# -*- coding: utf-8 -*-
"""AI 学习代理（v3）：让软件里配置的大模型**真正能操作软件**。

设计要点
- 工具即权限边界：模型只能调用本文件 `TOOLS` 里登记的 24 个函数，读写都强制带
  `user_id`，写入的 node_code 必须落在当前选中的那一门课里。
- 删除类工具（7 个）已开放：单条删除必须先查实 id；清空错题本一次调用完成。
  删除不可恢复，提示词里有明确守则（先核实、汇报删了什么）。
- 只读工具（10 个）任何情况下都开；写入/删除工具（14 个）受 `allow_write` 开关控制。
- 运行是显式状态机：act（模型+工具）→ review（质检员判定达标与否）→
  不达标回流 act（最多 2 次）→ finalize 出结论，防止「做一半就说完成」。
- 多轮 tool-calling 循环有硬上限（轮数 6 / 调用 16），防止模型打转烧额度。
- 所有工具返回值为 dict，出错也返回 dict（不抛异常打断对话），便于模型自我纠正。

为什么不放在 main.py：main 已经是上帝模块，这一层是纯逻辑，独立出来才能单测。
依赖通过 AgentContext 注入，避免和 main 循环导入。
"""

import json
import os
import re
import sqlite3
import time
import unicodedata
import uuid
from datetime import datetime, timedelta
from typing import Callable

from . import ai_helper, qushi

MAX_ROUNDS = int(os.getenv("AI_MAX_ROUNDS", "40"))  # 一轮对话最多几轮「模型→工具→模型」
MAX_TOOL_CALLS = int(os.getenv("AI_MAX_TOOL_CALLS", "150"))  # 一轮对话最多执行多少次工具
MAX_REVIEWS = int(os.getenv("AI_MAX_REVIEWS", "3"))  # 质量审查不达标最多回流几次，防审查-回流死循环
MAX_WALL_SECONDS = int(
    os.getenv("AI_MAX_WALL_SECONDS", "900")
)  # 单次对话总时长软上限（秒），防失控空转
MAX_TOOL_RESULT = 3000  # 单个工具结果回灌给模型的字符上限
MAX_MATERIAL_CHARS = 20000  # AI 生成资料入库的字数上限
MAX_QUESTIONS_PER_CALL = 10

# 出题工具的参数错误提示：必须带可照抄的示例，模型才能下一轮自己改对
_QUESTIONS_HINT = (
    "questions 必须是 JSON 数组（不是字符串），每项形如"
    ' {"type":"单选题","question":"题干","answer":"参考答案","evidence":"教材原文原句"}，'
    "type 和 answer 可省略，evidence 必填（没有原文依据的题不会入库）"
)

# navigate 工具可去的页面（前端路由白名单）
UI_ROUTES = {
    "/": "学习看板",
    "/syllabus": "教材目录",
    "/timeline": "四史时间轴",
    "/materials": "史料笔记",
    "/knowledge": "史实要点",
    "/tasks": "复习任务",
    "/practice": "自测",
    "/wrong-book": "错题本",
}
# 节 code 形如 socialist-01-02 / party-03
_NODE_CODE_RE = re.compile(r"^[a-z]+-\d{1,2}(?:-\d{1,2})*$")

# 单次对话各写入工具的批数上限（防"全流程狂写"烧光调用预算）
WRITE_BUDGETS = {
    "create_knowledge_point": 24,
    "create_questions": 16,
    "create_task": 20,
    "create_material": 6,
    "create_event": 6,
    "save_memory": 8,
    "highlight_text": 40,
    # 删除类：单条删除给足批量清理的余量；清空错题本一次搞定
    "delete_task": 40,
    "delete_knowledge_point": 40,
    "delete_material": 20,
    "delete_event": 20,
    "delete_question": 50,
    "delete_memory": 20,
    "clear_wrong_book": 1,
}

# 单次对话的读工具次数上限：模型找不到目标时会一篇篇往下翻，
# 真机里出现过「一个出题请求连读 12 篇资料、白烧 51 秒」的空转。
READ_BUDGETS = {
    "read_material": 12,
    "get_node_text": 30,
}

EXAM_TYPES = {
    "recall": "史实识记",
    "sequence": "时序定位",
    "argument": "史论结合",
    "apply": "现实联系",
}

# 错题归因的分类（移植 educational-tutor-ai 的「错因分析」思路，落到本项目字段上）。
# 只做「一类错因」而不是打分表：学生看错题时最需要知道的是「我下次该怎么改」。
ERROR_TYPES = {
    "concept": "概念混淆",
    "sequence": "时序错位",
    "relation": "史论脱节",
    "detail": "细节缺失",
    "reading": "审题偏差",
    "memory": "记忆不牢",
}


class AgentContext:
    """一次对话的上下文：谁、学哪门课、怎么连库。"""

    def __init__(
        self,
        uid: int,
        track: str,
        conn_factory: Callable[[], sqlite3.Connection],
        now_fn: Callable[[], str],
        page_context: str = "",
    ):
        self.uid = uid
        self.track = track
        self.conn = conn_factory
        self.now = now_fn
        self.page_context = (page_context or "").strip()[:200]  # 学生当前所在页面/正在做的事
        self.written: dict = {}  # 写入工具 -> 本对话成功写入的批数（预算控制用）
        self.reads: dict = {}  # 读工具 -> 本对话成功调用次数（防"通读所有资料"空转）

    @property
    def track_name(self) -> str:
        t = qushi.track_map().get(self.track)
        return t["name"] if t else ""

    def node_ok(self, code: str) -> bool:
        """章节必须存在，且属于当前课程（防止 AI 把党史内容写到别的课里）。"""
        if not code or not code.startswith(self.track + "-"):
            return False
        conn = self.conn()
        try:
            return qushi.node(conn, code) is not None
        finally:
            conn.close()

    def path_of(self, code: str) -> str:
        conn = self.conn()
        try:
            return qushi.path_text(conn, code)
        finally:
            conn.close()


def _clip(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "……"


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_CN_NUM = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _chapter_index(text: str):
    """把「第三章 / 第3章 / 3 / 三」归一成章序号；认不出返回 None。

    模型传 chapter 时经常给中文数字，而目录标题是「第三章 xxx」——
    直接字符串包含虽然能蒙对，但传「第3章」就失效了（会退化成全量目录，
    193 行被 6000 字截断，模型反而看不到目标章）。
    """
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"(\d{1,2})", t)
    if m:
        return int(m.group(1))
    m = re.search(r"第?([一二三四五六七八九十]{1,3})", t)
    if m:
        return _CN_NUM.get(m.group(1))
    return None


# ==================== 只读工具 ====================


def _tool_list_syllabus(ctx: AgentContext, args: dict) -> dict:
    """章节树；可按章名/章号收窄，避免一次吐 193 行。"""
    conn = ctx.conn()
    try:
        chapters = qushi.tree(conn, ctx.track, ctx.uid)
    finally:
        conn.close()
    want = (args.get("chapter") or "").strip()
    all_chapters = chapters
    if want:
        idx = _chapter_index(want)
        picked = []
        if idx:
            picked = [c for c in chapters if c["title"].startswith(f"第{qushi.chapter_no(idx)}章")]
        if not picked:
            picked = [
                c for c in chapters if want in c["title"] or want == c["code"] or want in c["code"]
            ]
        if not picked:
            names = [c["title"] for c in all_chapters]
            return {
                "ok": False,
                "error": f"没找到「{want}」对应的一章（当前课程共 {len(all_chapters)} 章）",
                "chapters": names,
                "hint": "从上面的章名里挑一个重试，或换个说法（如“第三章”/“第3章”）；"
                "需要完整目录时不传 chapter。",
            }
        chapters = picked
    lines = []
    for c in chapters:
        lines.append(f"{c['code']} {c['title']}（知识点 {c['kp']} / 题 {c['q']}）")
        for s in c["sections"]:
            lines.append(f"  {s['code']} {s['title']}（kp {s['kp']}）")
    return {
        "track": ctx.track,
        "track_name": ctx.track_name,
        "chapter_count": len(chapters),
        "text": "\n".join(lines)[:6000],
        "hint": "写入知识点/资料时请用上面的节 code（level 2 优先），不要用章 code。",
    }


def _tool_find_node(ctx: AgentContext, args: dict) -> dict:
    keyword = (args.get("keyword") or "").strip()
    if not keyword:
        return {"ok": False, "error": "keyword 不能为空"}
    conn = ctx.conn()
    try:
        rows = conn.execute(
            "SELECT code, level, title, period FROM syllabus_nodes "
            "WHERE track=? AND title LIKE ? ORDER BY level, order_no LIMIT 20",
            (ctx.track, f"%{keyword}%"),
        ).fetchall()
    finally:
        conn.close()
    return {
        "matched": len(rows),
        "nodes": [
            {
                "code": r["code"],
                "level": r["level"],
                "title": r["title"],
                "path": ctx.path_of(r["code"]),
            }
            for r in rows
        ],
        "hint": (
            "没匹配到就换更短的关键词（如「遵义」而不是「遵义会议的意义」）。" if not rows else ""
        ),
    }


def _tool_search_content(ctx: AgentContext, args: dict) -> dict:
    """全库检索：史料片段 + 知识点 + 时间轴事件，三类一起返回。"""
    query = (args.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query 不能为空"}
    limit = min(_as_int(args.get("limit"), 5) or 5, 8)
    conn = ctx.conn()
    try:
        mats = conn.execute(
            "SELECT id, title, content, node_code FROM materials WHERE user_id=?",
            (ctx.uid,),
        ).fetchall()
        scored = []
        for m in mats:
            for chunk in ai_helper.chunk_text(m["content"], 500):
                score = ai_helper.keyword_score(query, chunk)
                if score > 0:
                    scored.append((score, m["id"], m["title"], m["node_code"], chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen, snippets = set(), []
        for score, mid, title, code, chunk in scored:
            if mid in seen:
                continue
            seen.add(mid)
            snippets.append(
                {
                    "material_id": mid,
                    "title": title,
                    "node_code": code,
                    "excerpt": _clip(chunk, 420),
                }
            )
            if len(snippets) >= limit:
                break

        kps = [
            {
                "id": r["id"],
                "title": r["title"],
                "node_code": r["node_code"],
                "description": _clip(r["description"], 220),
            }
            for r in conn.execute(
                "SELECT id,title,node_code,description FROM knowledge_points "
                "WHERE user_id=? AND (title LIKE ? OR description LIKE ?) LIMIT 6",
                (ctx.uid, f"%{query}%", f"%{query}%"),
            )
        ]
        events = [
            {
                "id": r["id"],
                "title": r["title"],
                "year": r["year"],
                "node_code": r["node_code"],
                "digest": _clip(r["significance"] or r["impact"] or "", 160),
            }
            for r in conn.execute(
                "SELECT id,title,year,node_code,impact,significance FROM events "
                "WHERE user_id=? AND (title LIKE ? OR significance LIKE ? OR impact LIKE ?) LIMIT 6",
                (ctx.uid, f"%{query}%", f"%{query}%", f"%{query}%"),
            )
        ]
    finally:
        conn.close()
    return {
        "query": query,
        "materials": snippets,
        "knowledge_points": kps,
        "events": events,
        "hint": "引用史料时请说明出自哪篇（title + material_id）；资料里没有的内容要如实说「资料未涉及」。",
    }


def _tool_read_material(ctx: AgentContext, args: dict) -> dict:
    mid = _as_int(args.get("material_id"))
    if not mid:
        return {"ok": False, "error": "material_id 必须是数字"}
    offset = max(_as_int(args.get("offset"), 0) or 0, 0)
    length = min(_as_int(args.get("length"), 2500) or 2500, 6000)
    conn = ctx.conn()
    try:
        row = conn.execute(
            "SELECT id, title, content, node_code, filetype FROM materials WHERE id=? AND user_id=?",
            (mid, ctx.uid),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {"ok": False, "error": f"资料 {mid} 不存在或不属于当前用户"}
    total = len(row["content"] or "")
    body = (row["content"] or "")[offset : offset + length]
    return {
        "material_id": mid,
        "title": row["title"],
        "node_code": row["node_code"],
        "filetype": row["filetype"],
        "total_chars": total,
        "offset": offset,
        "text": body,
        "has_more": offset + length < total,
        "hint": (
            f"未完可再读 offset={offset + length}。" if offset + length < total else "已读完。"
        )
        + " 出题的原文依据请用 get_node_text 的教材正文，学生资料只作风格参考，不必逐篇通读。",
    }


def _tool_list_knowledge_points(ctx: AgentContext, args: dict) -> dict:
    node_code = (args.get("node_code") or "").strip()
    keyword = (args.get("keyword") or "").strip()
    limit = min(
        _as_int(args.get("limit"), 30) or 30, 30
    )  # 上限 30：大结果反复回灌会触发平台内容审查
    sql = "SELECT id,title,node_code,exam_type,description FROM knowledge_points WHERE user_id=?"
    params = [ctx.uid]
    if node_code:
        sql += " AND node_code LIKE ?"
        params.append(f"{node_code}%")
    else:
        sql += " AND node_code LIKE ?"
        params.append(f"{ctx.track}-%")
    if keyword:
        sql += " AND (title LIKE ? OR description LIKE ?)"
        params += [f"%{keyword}%", f"%{keyword}%"]
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    conn = ctx.conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return {
        "count": len(rows),
        "items": [
            {
                "id": r["id"],
                "title": r["title"],
                "node_code": r["node_code"],
                "exam_type": r["exam_type"],
                "description": _clip(r["description"], 160),
            }
            for r in rows
        ],
        "hint": "已有知识点不要重复创建；要出题请直接用这些 id。",
    }


def _tool_list_events(ctx: AgentContext, args: dict) -> dict:
    keyword = (args.get("keyword") or "").strip()
    y1, y2 = _as_int(args.get("year_from")), _as_int(args.get("year_to"))
    limit = min(_as_int(args.get("limit"), 30) or 30, 30)
    sql = "SELECT id,title,year,year_label,node_code,kind FROM events WHERE user_id=?"
    params = [ctx.uid]
    if keyword:
        sql += " AND (title LIKE ? OR significance LIKE ?)"
        params += [f"%{keyword}%", f"%{keyword}%"]
    if y1 is not None:
        sql += " AND year>=?"
        params.append(y1)
    if y2 is not None:
        sql += " AND year<=?"
        params.append(y2)
    sql += " ORDER BY year LIMIT ?"
    params.append(limit)
    conn = ctx.conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return {
        "count": len(rows),
        "items": [
            {
                "id": r["id"],
                "title": r["title"],
                "year": r["year_label"] or r["year"],
                "node_code": r["node_code"],
                "kind": r["kind"],
            }
            for r in rows
        ],
    }


def _tool_get_my_stats(ctx: AgentContext, args: dict) -> dict:
    conn = ctx.conn()
    try:
        totals = conn.execute(
            """SELECT (SELECT COUNT(*) FROM materials WHERE user_id=?) AS materials,
                      (SELECT COUNT(*) FROM knowledge_points WHERE user_id=?) AS kps,
                      (SELECT COUNT(*) FROM questions q JOIN knowledge_points k ON k.id=q.kp_id
                        WHERE k.user_id=?) AS questions,
                      (SELECT COUNT(*) FROM tasks WHERE user_id=? AND status!='done') AS open_tasks,
                      (SELECT COUNT(*) FROM practice_records WHERE user_id=?) AS attempts,
                      (SELECT COUNT(*) FROM wrong_book WHERE user_id=? AND resolved=0) AS unresolved_wrong""",
            (ctx.uid,) * 6,
        ).fetchone()
        per_kp = conn.execute(
            """SELECT k.id, k.title, COUNT(DISTINCT q.id) AS qn, COUNT(pr.id) AS attempts,
                      SUM(pr.correct) AS correct
               FROM knowledge_points k
               LEFT JOIN questions q ON q.kp_id=k.id
               LEFT JOIN practice_records pr ON pr.question_id=q.id
               WHERE k.user_id=? AND k.node_code LIKE ?
               GROUP BY k.id HAVING attempts>0 ORDER BY k.id""",
            (ctx.uid, f"{ctx.track}-%"),
        ).fetchall()
        due = conn.execute(
            "SELECT title, due_date FROM tasks WHERE user_id=? AND status!='done' "
            "AND due_date IS NOT NULL ORDER BY due_date LIMIT 5",
            (ctx.uid,),
        ).fetchall()
    finally:
        conn.close()
    weak = []
    for r in per_kp:
        acc = round((r["correct"] or 0) / r["attempts"] * 100, 1) if r["attempts"] else None
        if acc is not None and r["attempts"] >= 2 and acc < 60:
            weak.append(
                {
                    "kp_id": r["id"],
                    "title": r["title"],
                    "accuracy": acc,
                    "attempts": r["attempts"],
                }
            )
    return {
        "track": ctx.track_name,
        "totals": dict(totals),
        "weak_kps": weak,
        "upcoming_tasks": [{"title": d["title"], "due_date": d["due_date"]} for d in due],
        "hint": "给复习建议时请结合 weak_kps 与 unresolved_wrong，不要泛泛而谈。",
    }


def _tool_wrong_book(ctx: AgentContext, args: dict) -> dict:
    limit = min(_as_int(args.get("limit"), 10) or 10, 20)
    conn = ctx.conn()
    try:
        rows = conn.execute(
            """SELECT w.id, w.wrong_count, w.error_type, q.question, q.answer, q.type,
                      k.title AS kp_title, k.id AS kp_id
               FROM wrong_book w
               JOIN questions q ON q.id=w.question_id
               JOIN knowledge_points k ON k.id=q.kp_id
               WHERE w.user_id=? AND w.resolved=0 AND k.node_code LIKE ?
               ORDER BY w.last_wrong_at DESC LIMIT ?""",
            (ctx.uid, f"{ctx.track}-%", limit),
        ).fetchall()
    finally:
        conn.close()
    return {
        "count": len(rows),
        "error_types": ERROR_TYPES,
        "items": [
            {
                "wrong_id": r["id"],
                "kp_id": r["kp_id"],
                "kp_title": r["kp_title"],
                "type": r["type"],
                "question": _clip(r["question"], 200),
                "answer": _clip(r["answer"], 200),
                "wrong_count": r["wrong_count"],
                "error_type": r["error_type"] or "",
            }
            for r in rows
        ],
    }


# ==================== 写入工具 ====================


def _tool_create_knowledge_point(ctx: AgentContext, args: dict) -> dict:
    title = (args.get("title") or "").strip()
    if not title:
        return {"ok": False, "error": "title 不能为空"}
    node_code = (args.get("node_code") or "").strip() or ctx.track
    if node_code != ctx.track and not ctx.node_ok(node_code):
        return {
            "ok": False,
            "error": f"章节 {node_code} 不属于当前课程，请先用 find_node 查正确的节 code",
        }
    exam_type = (args.get("exam_type") or "").strip()
    if exam_type and exam_type not in EXAM_TYPES:
        exam_type = ""
    conn = ctx.conn()
    try:
        dup = conn.execute(
            "SELECT id FROM knowledge_points WHERE user_id=? AND node_code=? AND title=?",
            (ctx.uid, node_code, title),
        ).fetchone()
        if dup:
            return {
                "ok": True,
                "id": dup["id"],
                "duplicated": True,
                "message": f"已存在同名知识点（id={dup['id']}），未重复创建",
            }
        cur = conn.execute(
            "INSERT INTO knowledge_points"
            "(user_id,uuid,material_id,course,title,description,node_code,exam_type,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                ctx.uid,
                uuid.uuid4().hex,
                _as_int(args.get("material_id")),
                ctx.track,
                title,
                (args.get("description") or "").strip(),
                node_code,
                exam_type,
                ctx.now(),
            ),
        )
        conn.commit()
        kid = cur.lastrowid
    finally:
        conn.close()
    return {
        "ok": True,
        "id": kid,
        "node_code": node_code,
        "path": ctx.path_of(node_code),
        "message": f"已新建知识点「{title}」",
    }


def _tool_create_material(ctx: AgentContext, args: dict) -> dict:
    title = (args.get("title") or "").strip()
    content = (args.get("content") or "").strip()
    if not title or not content:
        return {"ok": False, "error": "title 与 content 都不能为空"}
    if len(content) < 80:
        return {"ok": False, "error": "正文太短（<80 字），史料要有实质内容"}
    if len(content) > MAX_MATERIAL_CHARS:
        return {
            "ok": False,
            "error": f"正文超过 {MAX_MATERIAL_CHARS} 字上限，请拆成多篇",
        }
    node_code = (args.get("node_code") or "").strip() or ctx.track
    if node_code != ctx.track and not ctx.node_ok(node_code):
        return {"ok": False, "error": f"章节 {node_code} 不属于当前课程"}
    conn = ctx.conn()
    try:
        dup = conn.execute(
            "SELECT id FROM materials WHERE user_id=? AND title=?", (ctx.uid, title)
        ).fetchone()
        if dup:
            return {
                "ok": True,
                "id": dup["id"],
                "duplicated": True,
                "message": f"已存在同名资料（id={dup['id']}），未重复创建",
            }
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        cur = conn.execute(
            "INSERT INTO materials(user_id,uuid,title,course,filename,filetype,content,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                ctx.uid,
                uuid.uuid4().hex,
                title,
                ctx.track,
                f"ai-{stamp}.md",
                "markdown",
                content,
                ctx.now(),
            ),
        )
        conn.commit()
        mid = cur.lastrowid
    finally:
        conn.close()
    return {
        "ok": True,
        "id": mid,
        "chars": len(content),
        "node_code": node_code,
        "message": f"已把《{title}》（{len(content)} 字）存入史料笔记",
    }


def _tool_get_node_text(ctx: AgentContext, args: dict) -> dict:
    """读教材正文（syllabus_nodes.content），可分页。出题前必须用它拿原文依据。"""
    node_code = (args.get("node_code") or "").strip()
    if not node_code:
        return {
            "ok": False,
            "error": "node_code 必填（来自 find_node / list_syllabus）",
        }
    offset = max(_as_int(args.get("offset"), 0) or 0, 0)
    length = min(_as_int(args.get("length"), 2500) or 2500, 6000)
    conn = ctx.conn()
    try:
        row = conn.execute(
            "SELECT code,title,content FROM syllabus_nodes WHERE code=?", (node_code,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        # 模型猜错 code 时会一路重试（真机里连错 2 次还接着猜），
        # 直接把同一章的真实节 code 摊开给它，让它一次改对
        avail = []
        m = re.match(r"^([a-z]+-\d{1,2})(?:-\d{1,2})*$", node_code)
        if m:
            conn = ctx.conn()
            try:
                avail = [
                    {"code": r["code"], "title": r["title"]}
                    for r in conn.execute(
                        "SELECT code,title FROM syllabus_nodes "
                        "WHERE code LIKE ? AND level=2 ORDER BY order_no LIMIT 40",
                        (m.group(1) + "-%",),
                    )
                ]
            finally:
                conn.close()
        return {
            "ok": False,
            "error": f"章节 {node_code} 不存在"
            + (
                f"，{m.group(1)} 下真实的节是：{', '.join(a['code'] for a in avail[:12])}"
                if avail
                else "，请先用 list_syllabus 确认节 code"
            ),
            "available": avail,
            "hint": "只能从 available 里挑 code，不要自己拼数字。",
        }
    content = row["content"] or ""
    if not content.strip():
        return {
            "ok": False,
            "error": f"「{row['title']}」这一节暂无教材正文，给不出原文依据，"
            "请换有正文的节出题，不要凭模型记忆编题",
        }
    text = content[offset : offset + length]
    return {
        "ok": True,
        "node_code": node_code,
        "title": row["title"],
        "text": text,
        "total_chars": len(content),
        "hint": "出题时每题的 evidence 必须从这里摘原句；写不出依据就不要给该知识点出题。",
    }


def _locate_quote(content: str, quote: str):
    """在正文里定位摘录句，返回 (offset, length)；找不到返回 None。

    两级容错：先去空白逐字匹配；不行再去掉标点/符号后匹配（模型摘录时
    经常把全角标点写成半角、或多/少一个引号）。用索引映射还原原文位置。
    """

    def _norm_indices(text, drop_punct):
        idxs, norm = [], []
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            if drop_punct:
                cat = unicodedata.category(ch)
                if cat.startswith("P") or cat.startswith("S"):
                    continue
            idxs.append(i)
            norm.append(ch)
        return idxs, "".join(norm)

    q = "".join(quote.split())
    if not (2 <= len(q) <= 200):
        return None
    for drop in (False, True):
        idxs, norm = _norm_indices(content, drop)
        _, qn = _norm_indices(quote, drop)
        if not qn:
            continue
        at = norm.find(qn)
        if at < 0:
            # 退化：模型常在句子中段改字/漏字（OCR 差异如「进人/进入」），
            # 取「最长能定位的前缀」；下界固定 12 字，短于此的不算有效依据
            lo = 12
            for cut in range(len(qn) - 1, lo - 1, -1):
                at = norm.find(qn[:cut])
                if at >= 0:
                    qn = qn[:cut]
                    break
            else:
                at = -1
        if at >= 0:
            offset = idxs[at]
            end = idxs[min(at + len(qn) - 1, len(idxs) - 1)] + 1
            if end - offset >= 2:
                return offset, end - offset
    return None


def _tool_highlight_text(ctx: AgentContext, args: dict) -> dict:
    """AI 划重点：把教材正文里的关键句高亮为 AI 专属紫色（text_notes, color=purple）。"""
    node_code = (args.get("node_code") or "").strip()
    quote = (args.get("quote") or "").strip()
    note = (args.get("note") or "").strip()
    if not node_code or not quote:
        return {
            "ok": False,
            "error": "node_code 和 quote 必填（quote 从 get_node_text 的正文里逐字摘录）",
        }
    if node_code != ctx.track and not ctx.node_ok(node_code):
        return {
            "ok": False,
            "error": f"章节 {node_code} 不属于当前课程，请先用 find_node 查正确的节 code",
        }
    q = "".join(quote.split())  # 忽略空白差异定位（与前端 locateQuote 同思路）
    if not (2 <= len(q) <= 200):
        return {"ok": False, "error": "quote 需 2~200 字，太长请只截取关键句"}
    conn = ctx.conn()
    try:
        row = conn.execute(
            "SELECT title,content FROM syllabus_nodes WHERE code=?", (node_code,)
        ).fetchone()
        if not row:
            return {"ok": False, "error": f"章节 {node_code} 不存在"}
        content = row["content"] or ""
        if not content.strip():
            return {
                "ok": False,
                "error": f"「{row['title']}」暂无教材正文，没有可划的重点",
            }
        hit = _locate_quote(content, quote)
        if hit is None:
            return {
                "ok": False,
                "error": "quote 没能在本节正文里逐字找到——必须原句摘录（不要改写、不要翻译），"
                "直接从 get_node_text 返回的正文复制（可只截前 16 字以上），不要自己加引号或改标点",
            }
        offset, length = hit
        now = ctx.now()
        dup = conn.execute(
            "SELECT id FROM text_notes WHERE user_id=? AND node_code=? AND offset=? AND length=?",
            (ctx.uid, node_code, offset, length),
        ).fetchone()
        if dup:
            conn.execute(
                "UPDATE text_notes SET note=?, color='purple', updated_at=? WHERE id=?",
                (note[:2000], now, dup["id"]),
            )
            nid, action = dup["id"], "updated"
        else:
            cur = conn.execute(
                "INSERT INTO text_notes(user_id,uuid,node_code,offset,length,quote,note,color,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    ctx.uid,
                    uuid.uuid4().hex,
                    node_code,
                    offset,
                    length,
                    quote[:600],
                    note[:2000],
                    "purple",
                    now,
                    now,
                ),
            )
            nid, action = cur.lastrowid, "created"
        conn.commit()
    finally:
        conn.close()
    return {
        "ok": True,
        "note_id": nid,
        "action": action,
        "node_code": node_code,
        "offset": offset,
        "length": length,
        "message": f"已在教材「{row['title']}」里划出重点（AI 专属紫色），学生打开这一节即可看到",
    }


_CHOICE_MARK = re.compile(r"(?:^|[^A-Za-z0-9])([A-D])\s*[.、．)）：:]\s*")


def _split_choice_options(text: str) -> list:
    """从题干里摘出 A/B/C/D 选项（与前端 parseOptions 同口径），返回 [(字母, 选项全文)]。

    找最长连续 A,B,C(,D) 段，至少 3 项才算选择题；摘不出返回空表。
    """
    marks = [(m.group(1), m.end(), m.start(1)) for m in _CHOICE_MARK.finditer(text)]
    best: list = []
    for i, (key, start, lead) in enumerate(marks):
        if key != "A":
            continue
        seq = [(key, start, lead)]
        for k2, s2, l2 in marks[i + 1 :]:
            if k2 == chr(65 + len(seq)):
                seq.append((k2, s2, l2))
            else:
                break
        if len(seq) >= 3 and len(seq) > len(best):
            best = seq
    if not best:
        return []
    return [
        (
            key,
            text[start : (best[i + 1][2] if i + 1 < len(best) else len(text))].strip(),
        )
        for i, (key, start, _) in enumerate(best)
    ]


def _normalize_choice_answer(qtype: str, question: str, answer: str) -> str:
    """选择题答案归一成「字母. 选项全文」，前端 answerKey 只认开头字母。

    模型可能给 "B"、"B. xxx"、整个选项原文、甚至同义改写（2026-11-10 实测：
    选项「和平与发展是时代主题」，答案给成「和平与发展是当代世界的两大问题」）——
    能对上选项的就换算成字母，对不上就原样返回（前端自动回退成自评题，不会误判）。
    """
    if not re.match(r"^[A-D](\s|[.、．)）：:]|$)", (answer or "").strip().upper()):
        opts = _split_choice_options(question)
        if opts:
            a = (answer or "").strip()
            best_key, best_score = None, 0.0
            for key, txt in opts:
                if a and (a in txt or txt in a or a in txt.replace("，", ",")):
                    return f"{key}. {txt}"
                # 模糊兜底：模型爱把选项改写一遍再当答案交上来。
                # ① 最长公共前缀 ≥4 字；② 字符重叠率 ≥0.7（对选项而言），取分高者。
                common = 0
                for c1, c2 in zip(a, txt):
                    if c1 == c2:
                        common += 1
                    else:
                        break
                if common >= 4:
                    score = 0.7 + 0.01 * common
                else:
                    tset, aset = set(txt), set(a)
                    score = len(tset & aset) / len(tset) if tset else 0.0
                if score >= 0.7 and score > best_score:
                    best_key, best_score = key, score
            if best_key:
                for key, txt in opts:
                    if key == best_key:
                        return f"{key}. {txt}"
    return answer


def _tool_create_questions(ctx: AgentContext, args: dict) -> dict:
    kp_id = _as_int(args.get("kp_id"))
    if not kp_id:
        return {
            "ok": False,
            "error": "kp_id 必须是数字，请先用 list_knowledge_points 查询",
        }
    items = args.get("questions") or []
    # 模型常见错误：把数组写成 JSON 字符串、或只传了一个对象 —— 这里都接住，
    # 别让一次格式抖动变成"重试 8 次全失败"（2026-11-07 真机评测发现的问题）
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            return {"ok": False, "error": _QUESTIONS_HINT}
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        return {"ok": False, "error": _QUESTIONS_HINT}
    if len(items) > MAX_QUESTIONS_PER_CALL:
        items = items[:MAX_QUESTIONS_PER_CALL]  # 超量截断，别整批打回重来
    conn = ctx.conn()
    try:
        kp = conn.execute(
            "SELECT id,title FROM knowledge_points WHERE id=? AND user_id=?",
            (kp_id, ctx.uid),
        ).fetchone()
        if not kp:
            return {"ok": False, "error": f"知识点 {kp_id} 不存在"}
        ids, skipped = [], []
        for i, q in enumerate(items):
            if not isinstance(q, dict):
                skipped.append(f"第{i + 1}项不是对象")
                continue
            question = str(q.get("question") or "").strip()
            if not question:
                skipped.append(f"第{i + 1}项缺 question")
                continue
            # 选择题：结构化 options 数组拼进题干（前端解析器只认题干内的 A. B. C. D.），
            # 题干里已有内联选项就不重复拼
            qtype = str(q.get("type") or "简答").strip()
            opts_arg = q.get("options")
            if isinstance(opts_arg, str):
                try:
                    opts_arg = json.loads(opts_arg)
                except json.JSONDecodeError:
                    opts_arg = None
            if isinstance(opts_arg, list):
                opts = [str(o).strip() for o in opts_arg if str(o).strip()][:4]
                if len(opts) >= 2 and not _CHOICE_MARK.search(question):
                    letters = "ABCD"
                    question = (
                        question.rstrip()
                        + "\n"
                        + "\n".join(f"{letters[i]}. {t}" for i, t in enumerate(opts))
                    )
            answer = str(q.get("answer") or "").strip()
            if "单选" in qtype or "选择" in qtype or "choice" in qtype.lower():
                answer = _normalize_choice_answer(qtype, question, answer)
            # 原文依据硬约束：模型给不出教材原句的题不落库（2026-11-09 用户要求）
            evidence = str(q.get("evidence") or "").strip()
            if not evidence:
                skipped.append(f"第{i + 1}项缺 evidence（教材原文依据）")
                continue
            cur = conn.execute(
                "INSERT INTO questions(user_id,uuid,kp_id,type,question,answer,evidence,source,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    ctx.uid,
                    uuid.uuid4().hex,
                    kp_id,
                    qtype,
                    question,
                    answer,
                    evidence[:200],
                    "ai",
                    ctx.now(),
                ),
            )
            ids.append(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()
        if not ids:
            return {
                "ok": False,
                "error": f"没有解析出可入库的题（{'；'.join(skipped[:3])}）。"
                "每题必须带 evidence（教材原文原句摘录），请先调 get_node_text 拿原文再出题。"
                + _QUESTIONS_HINT,
            }
    msg = f"已为「{kp['title']}」新增 {len(ids)} 道题"
    if skipped:
        msg += f"，另有 {len(skipped)} 条格式不对被跳过"
    return {
        "ok": True,
        "created": len(ids),
        "ids": ids,
        "skipped": skipped,
        "kp_title": kp["title"],
        "message": msg,
    }


def _parse_due(text: str, today: str = "") -> str | None:
    """宽容解析截止日期：ISO / 2026年11月12日 / 2026/11/12 / 明天 / 后天。

    模型经常拿不准日期格式（真机评测里 5 连败的根因），这里替它兜一层。
    解析不出返回 None，由调用方报错并附上今天的日期帮它换算。
    """
    s = (text or "").strip()
    if not s:
        return None
    try:
        base = datetime.fromisoformat(today).date() if today else datetime.now().date()
    except ValueError:
        base = datetime.now().date()
    rel = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}
    if s in rel:
        return (base + timedelta(days=rel[s])).isoformat()
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y年%m月%d日", "%Y年%m月", "%Y-%m"):
        for cand in (
            s,
            s.replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
            .replace("号", "")
            .replace("/", "-")
            .replace(".", "-"),
        ):
            try:
                return datetime.strptime(cand, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _tool_create_task(ctx: AgentContext, args: dict) -> dict:
    title = (args.get("title") or "").strip()
    if not title:
        return {"ok": False, "error": "title 不能为空"}
    today = (ctx.now() or "")[:10]
    due = _parse_due(args.get("due_date") or "", today)
    if args.get("due_date") and not due:
        return {
            "ok": False,
            "error": f"due_date 没认出来（收到「{args.get('due_date')}」）；"
            f"今天是 {today}，请换算成 2026-11-12 这样的具体日期",
        }
    conn = ctx.conn()
    try:
        cur = conn.execute(
            "INSERT INTO tasks(user_id,uuid,course,kp_id,title,description,due_date,status,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                ctx.uid,
                uuid.uuid4().hex,
                ctx.track,
                _as_int(args.get("kp_id")),
                title,
                (args.get("description") or "").strip(),
                due,
                "todo",
                ctx.now(),
            ),
        )
        conn.commit()
        tid = cur.lastrowid
    finally:
        conn.close()
    return {
        "ok": True,
        "id": tid,
        "due_date": due,
        "message": f"已创建复习任务「{title}」",
    }


def _tool_navigate(ctx: AgentContext, args: dict) -> dict:
    """UI 动作通道：让 AI 直接切换软件页面。

    后端只做白名单校验并把意图随事件流发回去，真正跳转由前端执行 ——
    学生说「切到史料笔记」，界面就直接切过去，不用再手动点侧栏。
    去教材页时还能带 node（节 code）+ quote（原文片段）：
    界面会翻到那一节并把这段原文临时标出来。
    """
    route = (args.get("route") or "").strip()
    if route not in UI_ROUTES:
        return {"ok": False, "error": f"route 必须是 {list(UI_ROUTES)} 之一"}
    ui = {"action": "navigate", "route": route}
    msg = f"已切换到「{UI_ROUTES[route]}」"
    node = (args.get("node") or "").strip().lower()
    if route == "/syllabus" and node:
        if not _NODE_CODE_RE.match(node):
            return {
                "ok": False,
                "error": "node 必须是真实节 code，例如 socialism-01-01",
            }
        query = {"node": node}
        quote = (args.get("quote") or "").strip()
        if quote:
            query["hl"] = quote[:80]
        ui["query"] = query
        msg = f"已翻到 {node} 并把这段原文标出来了" if quote else f"已翻到 {node}"
    return {
        "ok": True,
        "route": route,
        "label": UI_ROUTES[route],
        "ui": ui,
        "message": msg,
    }


def _tool_tag_wrong_error(ctx: AgentContext, args: dict) -> dict:
    """给一道错题打上错因标签（写 wrong_book.error_type）。

    移植自 educational-tutor-ai 的「错题归因」：先看学生错在哪、再给一类可操作的
    错因，而不是笼统说「这题不会」。归因由模型判断，本工具只负责校验与落库。
    """
    wid = _as_int(args.get("wrong_id"))
    raw = (args.get("error_type") or "").strip()
    if not wid or not raw:
        return {"ok": False, "error": "wrong_id 与 error_type 都必填"}
    key = raw.lower()
    label = ERROR_TYPES.get(key, raw)
    if label not in ERROR_TYPES.values():
        return {
            "ok": False,
            "error": f"error_type 只能取 {list(ERROR_TYPES.keys())} 或 {list(ERROR_TYPES.values())} 之一",
        }
    conn = ctx.conn()
    try:
        row = conn.execute(
            "SELECT id FROM wrong_book WHERE id=? AND user_id=?", (wid, ctx.uid)
        ).fetchone()
        if not row:
            return {"ok": False, "error": "错题记录不存在，或不属于当前用户"}
        conn.execute(
            "UPDATE wrong_book SET error_type=? WHERE id=? AND user_id=?",
            (label, wid, ctx.uid),
        )
        conn.commit()
    finally:
        conn.close()
    advice = (args.get("advice") or "").strip()
    return {
        "ok": True,
        "wrong_id": wid,
        "error_type": label,
        "message": f"错题 #{wid} 已归因为「{label}」" + (f"，建议：{advice}" if advice else ""),
    }


# ==================== 用户画像 / 长期记忆 ====================
# kind 取值：profile 画像（年级/考试时间等）、preference 偏好、weakness 薄弱点、plan 计划
MEMORY_KINDS = {
    "profile": "学生画像（年级、专业、考试时间等基本盘）",
    "preference": "学习偏好（喜欢的讲解方式、出题类型等）",
    "weakness": "薄弱点（反复出错的知识块）",
    "plan": "计划与约定（答应学生的复习安排等）",
}


def _tool_save_memory(ctx: AgentContext, args: dict) -> dict:
    """沉淀一条长期记忆，刻画用户画像，后续对话自动注入 system prompt。"""
    kind = (args.get("kind") or "profile").strip().lower()
    content = (args.get("content") or "").strip()
    if kind not in MEMORY_KINDS:
        return {"ok": False, "error": f"kind 只能取 {list(MEMORY_KINDS.keys())} 之一"}
    if not content:
        return {"ok": False, "error": "content 必填（一句话，300 字以内）"}
    content = content[:300]
    conn = ctx.conn()
    try:
        row = conn.execute(
            "SELECT id FROM user_memory WHERE user_id=? AND kind=? AND content=?",
            (ctx.uid, kind, content),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE user_memory SET updated_at=? WHERE id=?",
                (ctx.now(), row["id"]),
            )
            conn.commit()
            return {
                "ok": True,
                "id": row["id"],
                "duplicated": True,
                "message": "这条记忆已有，已刷新时间",
            }
        conn.execute(
            "INSERT INTO user_memory(user_id,uuid,kind,content,source,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (ctx.uid, uuid.uuid4().hex, kind, content, "agent", ctx.now(), ctx.now()),
        )
        conn.commit()
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM user_memory WHERE user_id=?", (ctx.uid,)
        ).fetchone()["n"]
    finally:
        conn.close()
    return {"ok": True, "message": f"已记住（{MEMORY_KINDS[kind]}），当前共 {n} 条"}


def _tool_list_memory(ctx: AgentContext, args: dict) -> dict:
    """查看已沉淀的记忆（按更新时间倒序）。"""
    kind = (args.get("kind") or "").strip().lower()
    limit = min(max(_as_int(args.get("limit")) or 20, 1), 50)
    conn = ctx.conn()
    try:
        if kind:
            rows = conn.execute(
                "SELECT id, kind, content, updated_at FROM user_memory "
                "WHERE user_id=? AND kind=? ORDER BY updated_at DESC LIMIT ?",
                (ctx.uid, kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, content, updated_at FROM user_memory "
                "WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                (ctx.uid, limit),
            ).fetchall()
    finally:
        conn.close()
    return {"ok": True, "items": [dict(r) for r in rows]}


# ---------------- 删除类工具（学生明确要求后开放；删除不可恢复） ----------------


def _delete_one(ctx: AgentContext, table: str, rid: int, label: str) -> dict:
    """单行删除的公共路径：必须带 id，只删本人数据，404 时提示先核实。"""
    if rid is None:
        return {
            "ok": False,
            "error": f"{label} 必须传 id（先调只读工具查到具体 id 再删）",
        }
    conn = ctx.conn()
    try:
        cur = conn.execute(f"DELETE FROM {table} WHERE id=? AND user_id=?", (rid, ctx.uid))
        conn.commit()
        if not cur.rowcount:
            return {
                "ok": False,
                "error": f"{label} #{rid} 不存在——先用只读工具核实 id，别凭记忆猜",
            }
        return {"ok": True, "deleted": rid, "message": f"已删除{label} #{rid}"}
    finally:
        conn.close()


def _tool_delete_task(ctx: AgentContext, args: dict) -> dict:
    return _delete_one(ctx, "tasks", _as_int(args.get("id")), "复习任务")


def _tool_delete_knowledge_point(ctx: AgentContext, args: dict) -> dict:
    return _delete_one(ctx, "knowledge_points", _as_int(args.get("id")), "知识点")


def _tool_delete_material(ctx: AgentContext, args: dict) -> dict:
    return _delete_one(ctx, "materials", _as_int(args.get("id")), "史料资料")


def _tool_delete_event(ctx: AgentContext, args: dict) -> dict:
    return _delete_one(ctx, "events", _as_int(args.get("id")), "时间轴事件")


def _tool_delete_memory(ctx: AgentContext, args: dict) -> dict:
    return _delete_one(ctx, "user_memory", _as_int(args.get("id")), "AI 记忆")


def _tool_delete_question(ctx: AgentContext, args: dict) -> dict:
    """删题并级联清掉它的作答记录与错题条目（与 DELETE /api/questions/{qid} 同逻辑）。"""
    qid = _as_int(args.get("id"))
    if qid is None:
        return {"ok": False, "error": "id 必填（先调只读工具查到题目 id）"}
    conn = ctx.conn()
    try:
        cur = conn.execute("DELETE FROM questions WHERE id=? AND user_id=?", (qid, ctx.uid))
        if not cur.rowcount:
            return {"ok": False, "error": f"题目 #{qid} 不存在——先核实 id"}
        conn.execute(
            "DELETE FROM practice_records WHERE question_id=? AND user_id=?",
            (qid, ctx.uid),
        )
        conn.execute("DELETE FROM wrong_book WHERE question_id=? AND user_id=?", (qid, ctx.uid))
        conn.commit()
        return {
            "ok": True,
            "deleted": qid,
            "message": f"已删除题目 #{qid}（含其作答记录与错题条目）",
        }
    finally:
        conn.close()


def _tool_clear_wrong_book(ctx: AgentContext, args: dict) -> dict:
    """清空错题本。only_unresolved=true 只清未掌握的。一次调用完成，不逐条删。"""
    only = bool(args.get("only_unresolved", False))
    conn = ctx.conn()
    try:
        sql = "DELETE FROM wrong_book WHERE user_id=?"
        if only:
            sql += " AND resolved=0"
        cur = conn.execute(sql, (ctx.uid,))
        conn.commit()
        n = cur.rowcount
        return {
            "ok": True,
            "cleared": n,
            "message": f"已清空错题本 {n} 条" + ("（未掌握的）" if only else ""),
        }
    finally:
        conn.close()


def _tool_create_event(ctx: AgentContext, args: dict) -> dict:
    title = (args.get("title") or "").strip()
    year = _as_int(args.get("year"))
    if not title or year is None:
        return {
            "ok": False,
            "error": "title 与 year 都必填（year 为整数，公元前用负数）",
        }
    node_code = (args.get("node_code") or "").strip() or ctx.track
    if node_code != ctx.track and not ctx.node_ok(node_code):
        return {"ok": False, "error": f"章节 {node_code} 不属于当前课程"}
    digest = (args.get("summary") or "").strip()
    conn = ctx.conn()
    try:
        dup = conn.execute(
            "SELECT id FROM events WHERE user_id=? AND title=?", (ctx.uid, title)
        ).fetchone()
        if dup:
            return {
                "ok": True,
                "id": dup["id"],
                "duplicated": True,
                "message": f"时间轴已有「{title}」，未重复添加",
            }
        cur = conn.execute(
            """INSERT INTO events(user_id,uuid,kp_id,title,era,year,year_end,year_label,place,people,
               cause,process,impact,significance,tags,kind,node_code,created_at)
               VALUES(?,?,NULL,?,?,?,NULL,'',?,'','','',?,?,'AI','event',?,?)""",
            (
                ctx.uid,
                uuid.uuid4().hex,
                title,
                ctx.track_name,
                year,
                (args.get("place") or "").strip(),
                digest,
                digest,
                node_code,
                ctx.now(),
            ),
        )
        conn.commit()
        eid = cur.lastrowid
    finally:
        conn.close()
    return {"ok": True, "id": eid, "message": f"已加入时间轴：{year} {title}"}


# ==================== 工具注册表 ====================


def _spec(name, desc, props, required=(), write=False):
    return {
        "name": name,
        "write": write,
        "spec": {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(required),
                },
            },
        },
    }


_S = lambda desc: {"type": "string", "description": desc}
_I = lambda desc: {"type": "integer", "description": desc}
_B = lambda desc: {"type": "boolean", "description": desc}

TOOLS = {
    "list_syllabus": dict(
        _spec(
            "list_syllabus",
            "列出当前课程的教材章节树（章 code + 节 code + 已有知识点数）。写数据前必须先确认节点 code。",
            {"chapter": _S("可选：章名关键词或章 code，如「遵义」「party-03」；留空列出全部")},
        ),
        fn=_tool_list_syllabus,
    ),
    "find_node": dict(
        _spec(
            "find_node",
            "按关键词在教材目录里定位章节，返回候选 code。适合「遵义会议在哪一节」这类问题。",
            {"keyword": _S("章节标题里的关键词，越短越好")},
            ["keyword"],
        ),
        fn=_tool_find_node,
    ),
    "search_content": dict(
        _spec(
            "search_content",
            "在学生的史料库、知识点、时间轴里做关键词检索，返回原文片段。回答任何史实问题前都应先调它。",
            {
                "query": _S("检索关键词"),
                "limit": _I("返回几篇资料片段，默认 5，最多 8"),
            },
            ["query"],
        ),
        fn=_tool_search_content,
    ),
    "read_material": dict(
        _spec(
            "read_material",
            "读取某一篇史料的正文（可分页）。做史料分析、出材料题前用它看原文。",
            {
                "material_id": _I("资料 id"),
                "offset": _I("从第几个字开始，默认 0"),
                "length": _I("读多少字，默认 2500，最多 6000"),
            },
            ["material_id"],
        ),
        fn=_tool_read_material,
    ),
    "get_node_text": dict(
        _spec(
            "get_node_text",
            "读教材正文（按节 code，可分页）。出题前必须先调它拿原文依据；"
            "回答知识点问题时也可以引用它作为权威原文。",
            {
                "node_code": _S("节 code，如 party-03-02"),
                "offset": _I("从第几个字开始，默认 0"),
                "length": _I("读多少字，默认 2500，最多 6000"),
            },
            ["node_code"],
        ),
        fn=_tool_get_node_text,
    ),
    "highlight_text": dict(
        _spec(
            "highlight_text",
            "在教材正文里划重点：把一段关键句高亮成 AI 专属紫色，学生打开该节正文立即可见。"
            "学生说「帮我画重点/标出重点/把重点标到教材上」时用——这是唯一能往教材上做标记的工具，"
            "把重点整理成史料笔记/资料**不能代替**它。quote 必须从 get_node_text 返回的正文里逐字摘录（不能改写），"
            "可附一句 note 批注说明为什么关键。",
            {
                "node_code": _S("节 code，如 party-03-02"),
                "quote": _S("要高亮的原文原句（2~200 字，逐字摘录）"),
                "note": _S("可选：一句批注，说明这句为什么重要"),
            },
            ["node_code", "quote"],
            write=True,
        ),
        fn=_tool_highlight_text,
    ),
    "list_knowledge_points": dict(
        _spec(
            "list_knowledge_points",
            "列出本人已有的知识点（可按节 code 或关键词过滤），返回 id 供出题使用。",
            {
                "node_code": _S("可选：节 code 前缀，如 party-03-02"),
                "keyword": _S("可选：标题/描述关键词"),
                "limit": _I("最多返回几条，默认 30"),
            },
        ),
        fn=_tool_list_knowledge_points,
    ),
    "list_events": dict(
        _spec(
            "list_events",
            "查时间轴事件，可按关键词与年份区间过滤。做时序题、梳理阶段特征时用。",
            {
                "keyword": _S("可选：事件名关键词"),
                "year_from": _I("可选：起始年"),
                "year_to": _I("可选：结束年"),
                "limit": _I("最多返回几条，默认 30"),
            },
        ),
        fn=_tool_list_events,
    ),
    "navigate": dict(
        _spec(
            "navigate",
            "直接切换软件页面（学生说「切到/打开/带我去 XX」时调用，界面会真的跳过去）。"
            "去教材页可带 node + quote：会翻到那一节并把这段原文临时标出来。",
            {
                "route": _S("页面路径：" + "，".join(f"{k}={v}" for k, v in UI_ROUTES.items())),
                "node": _S("可选：节 code（如 socialism-01-01），仅 /syllabus 有效"),
                "quote": _S("可选：要在教材里标出的原文片段（≤80 字，逐字摘自正文）"),
            },
            ["route"],
        ),
        fn=_tool_navigate,
    ),
    "get_my_stats": dict(
        _spec(
            "get_my_stats",
            "查看本人学习数据：资料/知识点/题目/任务数量、练习次数、错题数、薄弱知识点、临近任务。给复习建议前先调它。",
            {},
        ),
        fn=_tool_get_my_stats,
    ),
    "get_wrong_book": dict(
        _spec(
            "get_wrong_book",
            "看看未掌握的错题（题干、答案、所属知识点、已有的错因标签）。讲错题、做错因归因、定制复习计划前用。",
            {"limit": _I("最多返回几条，默认 10")},
        ),
        fn=_tool_wrong_book,
    ),
    "tag_wrong_error": dict(
        _spec(
            "tag_wrong_error",
            "给一道错题打错因标签，帮学生看清「错在哪一类」。做完归因再给针对性建议，别只打标签就完事。",
            {
                "wrong_id": _I("错题记录 id（来自 get_wrong_book 的 wrong_id）"),
                "error_type": _S(
                    "错因：concept 概念混淆 / sequence 时序错位 / relation 史论脱节 / "
                    "detail 细节缺失 / reading 审题偏差 / memory 记忆不牢"
                ),
                "advice": _S("可选：一句话说明下次怎么避免这类错"),
            },
            ["wrong_id", "error_type"],
            write=True,
        ),
        fn=_tool_tag_wrong_error,
    ),
    "create_knowledge_point": dict(
        _spec(
            "create_knowledge_point",
            "新建一条史实要点（知识点）并挂到某一节下。写之前先用 list_knowledge_points 查重。",
            {
                "node_code": _S("节 code，如 party-03-02（必填，先用 find_node 查）"),
                "title": _S("知识点标题，20 字以内"),
                "description": _S("要点正文：背景/内容/意义，150-400 字，按权威简史表述"),
                "exam_type": _S(
                    "考法，可选 recall(史实识记)/sequence(时序定位)/argument(史论结合)/apply(现实联系)"
                ),
                "material_id": _I("可选：关联的史料 id"),
            },
            ["node_code", "title", "description"],
            write=True,
        ),
        fn=_tool_create_knowledge_point,
    ),
    "create_material": dict(
        _spec(
            "create_material",
            "生成一篇史料/学习资料并存入「史料笔记」，学生之后能直接打开阅读、用来出题。适合整理原文要点、专题综述、会议文件摘编。",
            {
                "title": _S("资料标题"),
                "content": _S("正文，Markdown 格式，200-5000 字"),
                "node_code": _S("可选：挂到哪一节，默认挂当前课程"),
            },
            ["title", "content"],
            write=True,
        ),
        fn=_tool_create_material,
    ),
    "create_questions": dict(
        _spec(
            "create_questions",
            "给某个知识点批量新建练习题。questions 必须是 JSON 数组（不是字符串），"
            "每项都要有 question、answer 和 evidence（教材原文依据，摘自 get_node_text 的正文原句）；"
            "没有 evidence 的题不会入库。单选题：type 填「单选题」，options 传选项数组（题干里不要写选项），"
            "answer 填正确选项的字母（如 B）；其余题型（名词解释/时间排序/简答/材料分析/论述）不用 options。",
            {
                "kp_id": _I("知识点 id（来自 list_knowledge_points）"),
                "questions": {
                    "type": "array",
                    "description": '题目数组，最多 10 项；单选题如 {"type":"单选题","question":"…","options":["…","…","…","…"],"answer":"B","evidence":"…"}，简答如 {"type":"简答","question":"…","answer":"…","evidence":"…"}',
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": _S("题型，缺省为简答；单选题必须用「单选题」"),
                            "question": _S("题干（单选题不要把选项写进题干）"),
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "单选题专用：选项数组，2-4 个字符串，按 A/B/C/D 顺序",
                            },
                            "answer": _S("参考答案；单选题填正确选项字母（如 B）"),
                            "evidence": _S(
                                "教材原文依据（从 get_node_text 的正文里摘原句，≤80 字）"
                            ),
                        },
                        "required": ["question", "answer", "evidence"],
                    },
                },
            },
            ["kp_id", "questions"],
            write=True,
        ),
        fn=_tool_create_questions,
    ),
    "create_task": dict(
        _spec(
            "create_task",
            "给学生建一条复习任务（可带截止日期），会出现在「复习任务」页。",
            {
                "title": _S("任务标题"),
                "description": _S("任务说明：做什么、做到什么程度"),
                "due_date": _S("截止日期，形如 2026-11-12；可留空"),
                "kp_id": _I("可选：关联知识点 id"),
            },
            ["title"],
            write=True,
        ),
        fn=_tool_create_task,
    ),
    "create_event": dict(
        _spec(
            "create_event",
            "往时间轴补一条历史事件（会议/事件/文件）。学生问「能不能补进时间轴」或内容缺失时用。",
            {
                "title": _S("事件名称"),
                "year": _I("年份整数，公元前用负数"),
                "summary": _S("一句话史实要点（30-120 字）"),
                "place": _S("可选：地点"),
                "node_code": _S("可选：挂到哪一节"),
            },
            ["title", "year"],
            write=True,
        ),
        fn=_tool_create_event,
    ),
    "save_memory": dict(
        _spec(
            "save_memory",
            "把值得长期记住的学生信息存入记忆库（刻画用户画像，之后每次对话自动带上）。"
            "适合记：考试时间、专业背景、偏好、反复暴露的薄弱点、答应过的安排。"
            "只记稳定信息，不要把一次性的聊天内容存进来。",
            {
                "kind": _S("profile(画像)/preference(偏好)/weakness(薄弱点)/plan(计划)"),
                "content": _S("一句话记忆，300 字以内"),
            },
            ["kind", "content"],
            write=True,
        ),
        fn=_tool_save_memory,
    ),
    "list_memory": dict(
        _spec(
            "list_memory",
            "查看已存的学生记忆（画像/偏好/薄弱点/计划），制定复习计划前先看看。",
            {"kind": _S("可选：按类型过滤"), "limit": _I("最多返回几条，默认 20")},
        ),
        fn=_tool_list_memory,
    ),
    "delete_task": dict(
        _spec(
            "delete_task",
            "删除一条复习任务（不可恢复）。id 必须先用只读工具查实。",
            {"id": _I("任务 id")},
            ["id"],
            write=True,
        ),
        fn=_tool_delete_task,
    ),
    "delete_knowledge_point": dict(
        _spec(
            "delete_knowledge_point",
            "删除一条知识点（不可恢复；挂在其下的题目不受影响）。id 必须先查实。",
            {"id": _I("知识点 id")},
            ["id"],
            write=True,
        ),
        fn=_tool_delete_knowledge_point,
    ),
    "delete_material": dict(
        _spec(
            "delete_material",
            "删除一条史料/资料（不可恢复）。id 必须先查实。",
            {"id": _I("资料 id")},
            ["id"],
            write=True,
        ),
        fn=_tool_delete_material,
    ),
    "delete_event": dict(
        _spec(
            "delete_event",
            "删除一条时间轴事件（不可恢复）。id 必须先查实。",
            {"id": _I("事件 id")},
            ["id"],
            write=True,
        ),
        fn=_tool_delete_event,
    ),
    "delete_question": dict(
        _spec(
            "delete_question",
            "删除一道题目（不可恢复；其作答记录与错题条目一并删除）。id 必须先查实。",
            {"id": _I("题目 id")},
            ["id"],
            write=True,
        ),
        fn=_tool_delete_question,
    ),
    "delete_memory": dict(
        _spec(
            "delete_memory",
            "删除一条 AI 记忆（学生说「忘掉/删掉这条记忆」时用）。",
            {"id": _I("记忆 id")},
            ["id"],
            write=True,
        ),
        fn=_tool_delete_memory,
    ),
    "clear_wrong_book": dict(
        _spec(
            "clear_wrong_book",
            "清空错题本（一次调用全清，不要逐条删）。学生说「清空错题本/错题全删」时用。",
            {"only_unresolved": _B("只清未掌握的？默认 false 全清")},
            write=True,
        ),
        fn=_tool_clear_wrong_book,
    ),
}


def tool_specs(allow_write: bool) -> list:
    return [t["spec"] for t in TOOLS.values() if allow_write or not t["write"]]


def execute_tool(ctx: AgentContext, name: str, args: dict, allow_write: bool) -> dict:
    tool = TOOLS.get(name)
    if not tool:
        return {"ok": False, "error": f"没有名为 {name} 的工具"}
    if tool["write"] and not allow_write:
        return {
            "ok": False,
            "error": "当前会话关闭了写入权限，只能查不能改；请提示用户在界面上打开「允许 AI 写入」",
        }
    # 单次对话写入预算：模型一旦进入"全流程狂写"模式，会把调用预算烧光，
    # 反而没执行学生点名要的那个动作（真机评测 #4 两次翻车的根因）
    cap = WRITE_BUDGETS.get(name)
    if cap is not None and ctx.written.get(name, 0) >= cap:
        return {
            "ok": False,
            "error": f"本轮对话{name}已成功 {ctx.written[name]} 批（上限 {cap}），"
            "先收尾向学生汇报；他需要更多会再开口",
        }
    # 读工具同样限次：找不到目标时模型容易一篇篇往下翻，读得越多越难收尾
    rcap = READ_BUDGETS.get(name)
    if rcap is not None and ctx.reads.get(name, 0) >= rcap:
        return {
            "ok": False,
            "error": f"本轮对话{name}已读 {rcap} 次（上限）。已经拿到不少材料了，"
            "请基于现有内容作答或提问澄清，不要再继续翻。",
        }
    try:
        result = tool["fn"](ctx, args or {})
    except Exception as e:  # 工具内部异常不该打断整轮对话，回给模型让它换参数重试
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if cap is not None and result.get("ok"):
        ctx.written[name] = ctx.written.get(name, 0) + 1
    if rcap is not None and result.get("ok") is not False:  # 部分读工具成功时不带 ok 字段
        ctx.reads[name] = ctx.reads.get(name, 0) + 1
    return result


_WEEKDAYS = "一二三四五六日"


def _today_line() -> str:
    """提示词里的今天：让模型能自己算「明天/下周」。"""
    now = datetime.now()
    return f"{now:%Y-%m-%d} 星期{_WEEKDAYS[now.weekday()]}"


def _page_context_line(ctx: AgentContext) -> str:
    """学生所在功能页的提示词片段（抽出为函数，避免 f-string 内含反斜杠，兼容 py3.11）。"""
    if not ctx.page_context:
        return ""
    return (
        "\n学生此刻在「" + ctx.page_context + "」功能页内和你对话，回答尽量贴合这个场景"
        "（比如在自测页就多谈做题安排，在任务页就围绕复习计划）；"
        "他的指令若与页面相关，优先直接执行并顺势 navigate。"
    )


def _memory_line(ctx: AgentContext) -> str:
    """把已沉淀的用户画像注入 system prompt（最多 10 条，防 prompt 膨胀）。"""
    try:
        conn = ctx.conn()
        rows = conn.execute(
            "SELECT kind, content FROM user_memory WHERE user_id=? "
            "ORDER BY updated_at DESC LIMIT 10",
            (ctx.uid,),
        ).fetchall()
        conn.close()
    except Exception:
        return ""
    if not rows:
        return ""
    items = "\n".join(
        f"- [{MEMORY_KINDS.get(r['kind'], r['kind'])[:4]}] {r['content']}" for r in rows
    )
    return (
        f"\n已记住的学生信息（画像/偏好/薄弱点/计划，给建议时优先贴合）：\n{items}\n"
        "若对话中出现了新的稳定信息（考试时间、明显薄弱的知识块、明确偏好），"
        "顺手用 save_memory 沉淀一条；不要重复记录已有的。"
    )


def system_prompt(ctx: AgentContext) -> str:
    return f"""你是「四史学习助手」内置的学科助教兼学习管家，正在辅导一名大学生学习《{ctx.track_name}》。
你不是只会聊天的问答机器人：你能通过工具**直接读写这个软件里的数据**——检索教材目录与史料原文、建知识点、生成资料入库、出题、建复习任务、补时间轴。

工作方式（自主执行，不要反问一堆再动手）：
1. 涉及具体史实、章节、原文的问题，先调 search_content / find_node / list_syllabus 核实，再回答。
2. 学生说「帮我整理/生成/入库」时，直接把结果写进库（create_knowledge_point / create_material / create_questions / create_task），写完用一句话说明写了什么、在哪一页能看到。**整理资料类任务必须先用 search_content 查一遍学生自己的史料库**——已上传的笔记/资料是第一素材（用 read_material 读全文、基于它整理，并在答复里注明出处 material_id）；学生资料里没有的部分才用教材正文补，绝不要无视学生已上传的资料、直接从教材另起炉灶。若已存在相似主题的笔记，默认基于原文增补或明确说明「新写了什么、和旧笔记的区别」，不要默默重复入库。**整理资料类任务检索 1-2 次拿到素材就写**，写完就收尾，不要反复检索。
3. 学生说「帮我学第 X 章」时，按「查目录 → 检索已有内容 → 补知识点 → 出 2-3 道题 → 建复习任务」的顺序自主跑完，最后给一个三步以内的行动建议。
4. 写作前先用 list_knowledge_points 查重，避免重复入库；一次别超过 10 条知识点或 10 道题。
5. 互不依赖的工具在同一轮里一起调用（例如同时查目录和检索史料），别一个工具磨一轮；正常请求应在 2-3 轮内给结论。
6. 工具参数必须是**合法 JSON 对象**：数组字段（如 create_questions 的 questions）要传真正的 JSON 数组，不能是字符串；参数报错时看清错误里的示例，改对再调，不要原样重试。
7. 你一共有 8 轮、20 次调用。**长任务（学整章、建一套题）读类工具合计最多 3 次**：定位章节用 find_node（比 list_syllabus 快），检索一次就够；省下的轮次一定用来完成写入（建知识点→出题→建任务），宁可少而完整，不要读得多、写得少。
8. 写入有预算：知识点 6 条、出题 4 批、任务 5 条、资料 2 篇。**学生点名的动作优先**——比如他要的是复习任务，就先建任务；到预算上限就收尾汇报，别恋战。
9. 学生问「我为什么老错」或让你分析错题时：先 get_wrong_book 看题，逐题判断错因并调 tag_wrong_error 落库（概念混淆/时序错位/史论脱节/细节缺失/审题偏差/记忆不牢），最后按错因归并成 2-3 条改法。归因要基于题目本身，看不出就写「记忆不牢」，别硬编。
10. 学生要求处理**当前课程以外**的内容（比如你在教《党史》他却要新中国史的知识点/题目）：**不要借当前课程的名义写入**，直接提醒他到软件右上角「切换课程」再回来找你，一次说清即可。
11. 你能直接操控界面：学生说「切到史料笔记/打开错题本/带我去自测」就调 navigate；写完知识点、题目、任务、资料后主动调 navigate 把学生带到对应页面（知识点→/knowledge，题目→/practice，任务→/tasks，资料→/materials，事件→/timeline），并在答复里说「已经帮你切过去了」。
12. **出题只入库、不在聊天里做题**：调完 create_questions 后，最终答复**只说生成了几道题、覆盖什么考点**，绝不要把题目题干、选项、答案罗列进对话——界面会自动跳到自测页并直接开始作答，作答数据自动进错题本。聊天里贴题会把学生困在对话里做题，数据不落库，是严重错误。
13. 收到「【质量审查】」提示时，说明你上一稿被判没达标：只针对缺口补齐（可以再用工具），不要推翻已完成的有效工作，补完重新总结收尾。
14. 对话中出现值得长期记住的学生信息（考试时间、专业背景、反复暴露的薄弱块、明确偏好），顺手调 save_memory 沉淀；已有的信息不要重复存。
15. **删除守则**：删除不可恢复。单条删除（delete_*）前必须先用只读工具确认那条数据存在、确实是要删的那条，绝不凭记忆猜 id；删完后在答复里逐条列出删了什么。批量清理优先用一键工具（clear_wrong_book），不要一条一条删；学生说「全部删掉/清空」时，答复里先报告将删的数量再执行，结果如实汇报条数。
16. **词条卡**：学生问「讲讲 X / X 是什么 / 介绍 X / X 的中心思想」这类**单个知识点/事件/史料**的问题，且你查到了原文时，严格按下面的词条格式回答（前端会渲染成卡片；各部分的方括号、记号必须原样保留）：
# 标题
出处：{ctx.track}-01-01 第一章 空想社会主义的产生和发展
> 这里照录教材正文原文（可多段，整段照录不改写）
【要点】短标签一 · 短标签二 · 短标签三（3-5 个，每个 2-6 字）
**中心思想**：两三句概括它的核心内容与意义
—— 来源：《{ctx.track_name}》{ctx.track}-01-01 第一章 空想社会主义的产生和发展
（出处行和来源行都按上面两行的样子写：**节 code 在最前面**，后面跟章名/节名，不要写别的说明）
**硬性要求**：原文节选必须**逐字复制自 get_node_text 返回的教材正文**（或该节史料原文），出处行与来源行都必须写**真实节 code**——学生点卡片会直接跳到教材那一节并标出这段原文，code 错了或写了材料 id 就跳不了；引用的内容没有对应的教材节时不要套词条卡格式，正常分点回答。学生说「带我去原文/让我看看书上怎么写的」时，直接调 navigate(route="/syllabus", node=节 code, quote=原文片段)，界面会当场翻到那一节并标出来。
问题不是单个知识点、或没查到原文时不必套用，正常分点回答即可。
19. **模糊/情绪型请求先给结论**：学生只表达焦虑或没有明确任务（「下周就考试了很慌怎么办」「我该怎么复习」），先 get_my_stats 看一眼进度，直接给 3 步以内的具体做法收尾，最多建 2-3 条任务；**不要**借此一口气灌知识点、批量出题——他真要整理/出题会说清楚的。
17. **出题必须有原文依据**：调 create_questions 前必须先 get_node_text 读该知识点的节正文；每道题的 evidence 字段填教材正文里的**原句摘录**（≤80 字）；写不出依据的知识点不要给它出题，这一节没有教材正文就直接告诉学生出不了并说明原因，绝不凭模型记忆编题。学生说「出选择题/几道选择题练练」时优先出单选题：type 填「单选题」、options 传 4 个选项数组、answer 填字母；干扰项要有真实迷惑性——用相近年份、相近会议/人物、张冠李戴的表述做干扰，不要「以上都不对」这类废选项；其余情况默认简答/材料分析为主、单选题穿插 1-2 道。
18. **划重点**：学生请你「画重点/标重点/把重点标到教材上」时，先 get_node_text 读正文，再调 highlight_text 把真正关键的句子划成紫色——每节挑 2~5 处、每处 2~200 字，note 用一句话写清为什么关键；quote **直接从 get_node_text 返回的正文复制**（逐字，不要自己加引号/改标点/缩写；只截取前 16 字以上也能定位）。**绝不要用 create_material 把重点摘编存成资料来代替划重点**——那不是学生要的。答复里说明划了哪几处、各为什么关键。

内容要求：
- 口径必须符合权威简史（《百年奋斗重大成就和历史经验的决议》、四本官方简史、二十大报告），重大年份、会议、文件名称不得写错。
- 严格区分「史料原文」与「史家评价」；资料里没有的内容明确说「资料未涉及」，不得编造。
- 只谈《{ctx.track_name}》范围；学生问别的课程，提醒他到「切换课程」，不要混着讲。
- 回答用中文，先给结论，再分点展开；不超过 500 字，需要长文时先给框架。

当前课程：{ctx.track_name}（track={ctx.track}）。章节 code 形如 {ctx.track}-03-02，写入必须用真实的节 code。
今天是 {_today_line()}。学生说「明天/下周/这几天」这类相对日期时，先换算成具体日期再调工具。{_memory_line(ctx)}{_page_context_line(ctx)}"""


def _force_final(convo: list) -> str:
    """强制模型不带工具收尾。

    step-3.5-flash 在长上下文里偶尔返回空 content（答案落进 reasoning_content），
    这里补一次「直接回答」，还是空就退而取推理文本，保证学生不会看到空气泡。
    """
    try:
        msg = ai_helper.chat_completion(
            convo
            + [
                {
                    "role": "user",
                    "content": "请不要再调用工具，直接用中文给出最终答复：写入了什么、学生下一步做什么。",
                }
            ],
            tools=None,
        )
        text = (msg.get("content") or "").strip()
        if not text:
            text = (msg.get("reasoning_content") or "").strip()[:1500]
        return text
    except Exception:
        return ""


def _review_draft(user_goal: str, steps: list, draft: str) -> dict:
    """质检员：独立一次大模型调用，判断草稿答复是否已满足学生请求。

    返回 {"done": bool, "gap": str}。解析失败/调用失败一律按达标放行——
    审查是增强项，绝不能因为它把主流程卡死。
    """
    step_lines = (
        "\n".join(f"- {s['tool']}：{s['summary']}" for s in steps[-12:]) or "（未调用工具）"
    )
    judge = [
        {
            "role": "system",
            "content": (
                "你是学习助手的质检员。判断助教的答复是否已满足学生的请求。"
                '只输出一个 JSON 对象：{"done": true 或 false, "gap": "没做到的点，一句话"}。'
                "判定标准宁可宽松：史实答对、学生点名的动作（写入/建任务/出题等）确实执行了就算达标；"
                "格式瑕疵、措辞差异不算不达标。"
            ),
        },
        {
            "role": "user",
            "content": f"学生请求：{user_goal}\n\n已执行步骤：\n{step_lines}\n\n拟答复：\n{draft[:1200]}",
        },
    ]
    try:
        msg = ai_helper.chat_completion(judge, tools=None)
        text = (msg.get("content") or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
        return {
            "done": bool(data.get("done", True)),
            "gap": str(data.get("gap", ""))[:300],
        }
    except Exception:
        return {"done": True, "gap": ""}


def _is_censorship(e: Exception) -> bool:
    s = str(e).lower()
    return "451" in s or "censorship" in s or "blocked" in s


def _friendly_error(e: Exception) -> str:
    """把大模型接口报错翻译成人话：审查拦截是学生最容易碰到的，别吓人。"""
    s = str(e)
    low = s.lower()
    if "451" in s or "censorship" in low or "blocked" in low:
        return (
            "AI 平台的安全审查拦下了这轮内容（censorship_blocked）。这不是软件坏了——"
            "通常是单次请求里带的正文太长，请换个问法：拆成小问题、少让 AI 铺开大段原文，稍后重试即可。"
        )
    if "429" in s or "rate limit" in low:
        return "AI 平台限流了，等几秒再发一次就好。"
    if "timeout" in low or "timed out" in low:
        return "AI 平台响应超时，请重试一次。"
    return f"调用大模型失败：{e}"


def run_events(ctx: AgentContext, messages: list, allow_write: bool = True):
    """生成器：边跑边吐事件，最后一个事件是 {"type":"done", ...}。

    状态机（类 OpenClaw 的「感知→行动→审查→回流」）：
        plan/act  ：模型带着工具自主执行（一轮或多轮「模型→工具→模型」）
        review    ：模型停止调工具后，质检员独立判定答复是否满足请求
        act(回流) ：不达标且还有预算 → 把审查意见作为新一轮输入，回到 act
        finalize  ：达标 / 审查次数用尽 / 轮次用尽 → 出最终答复

    事件类型：
    - state     ：状态机切换（state=act/review，reason=回流原因）
    - step/start：开始调某个工具（前端先亮一张「正在检索史料」的卡片）
    - step/end  ：工具跑完（带 ok、一行 summary、完整 detail）
    - error     ：大模型调用失败
    - done      ：最终答复 + 完整步骤记录
    """
    specs = tool_specs(allow_write)
    convo = [{"role": "system", "content": system_prompt(ctx)}]
    # 学生最近一条请求：审查员要以它为基准判定达标与否
    user_goal = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user" and m.get("content")),
        "",
    )[:1500]
    for m in messages[-12:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            convo.append({"role": role, "content": content[:4000]})

    steps = []
    used = 0
    reviews = 0
    failures: dict = {}  # (tool, 参数) -> 上次失败原因：原样重试没有意义，直接短路
    succeeded: set = set()  # 已成功过的 (tool, 参数)：结果还在上下文里，重复调用只会白灌一遍大文本
    salvaged = False  # 审查拦截后「摘原文重试」只做一次，防救不回来时白耗轮次
    t_start = time.monotonic()
    stop_reason = "rounds"
    yield {"type": "state", "state": "act", "round": 1}
    for round_no in range(1, MAX_ROUNDS + 1):
        if time.monotonic() - t_start > MAX_WALL_SECONDS:
            stop_reason = "time"
            break
        try:
            msg = ai_helper.chat_completion(convo, tools=specs)
        except Exception as e:
            # 平台审查拦截多半是大段工具原文（党史长文）撑的：把原文摘掉换成一句话，
            # 保留工作记录再试一次，让前面几轮工具不至于白跑（只救一次）。
            if _is_censorship(e) and not salvaged and any(m.get("role") == "tool" for m in convo):
                salvaged = True
                for m in convo:
                    if m.get("role") == "tool":
                        m["content"] = (
                            "（工具返回的原文因平台内容审查被省略；"
                            "你的操作记录还在，请基于已有信息直接答复学生，不要再复读原文）"
                        )
                yield {
                    "type": "state",
                    "state": "act",
                    "reason": "审查拦截，已摘除大段原文重试",
                    "round": round_no,
                }
                continue
            yield {"type": "error", "message": _friendly_error(e)}
            yield {
                "type": "done",
                "reply": "",
                "steps": steps,
                "rounds": round_no,
                "tool_calls": used,
                "error": str(e),
            }
            return

        calls = msg.get("tool_calls") or []
        # 模型的中间想法：有工具调用时它的 content 就是「我打算先查什么」——实时吐给前端展示
        thinking = (msg.get("content") or "").strip()
        if calls and thinking:
            yield {"type": "thinking", "round": round_no, "text": thinking[:400]}
        if not calls:
            text = (msg.get("content") or "").strip()
            if not text:
                text = _force_final(convo)
            # ---- REVIEW：动过工具才有审查价值（纯问答直接放行，省一次调用）----
            if used > 0 and reviews < MAX_REVIEWS:
                reviews += 1
                yield {
                    "type": "state",
                    "state": "review",
                    "round": round_no,
                    "review": reviews,
                }
                verdict = _review_draft(user_goal, steps, text)
                if not verdict["done"]:
                    yield {
                        "type": "state",
                        "state": "act",
                        "reason": verdict["gap"],
                        "review": reviews,
                        "round": round_no,
                    }
                    convo.append({"role": "assistant", "content": text})
                    convo.append(
                        {
                            "role": "user",
                            "content": (
                                f"【质量审查】你上一稿没有完全满足学生的请求，缺口：{verdict['gap']}。"
                                "请继续完成目标（可以再用工具），完成后重新总结，"
                                "不要重复已经做过的事。"
                            ),
                        }
                    )
                    continue
            # ---- FINALIZE ----
            yield {"type": "state", "state": "finalize", "round": round_no}
            yield {
                "type": "done",
                "reply": text,
                "steps": steps,
                "rounds": round_no,
                "tool_calls": used,
                "reviews": reviews,
                "elapsed_ms": int((time.monotonic() - t_start) * 1000),
            }
            return

        convo.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": calls,
            }
        )
        for tc in calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            desc = TOOLS.get(name, {}).get("spec", {}).get("function", {}).get("description", "")
            yield {
                "type": "step",
                "phase": "start",
                "tool": name,
                "round": round_no,
                "label": desc[:20],
                "args": args,
                "write": bool(TOOLS.get(name, {}).get("write")),
            }
            t_tool = time.monotonic()

            if used >= MAX_TOOL_CALLS:
                result = {
                    "ok": False,
                    "error": "工具调用次数已达上限，请直接基于已有信息作答",
                }
            else:
                key = (
                    name,
                    (
                        raw_args
                        if isinstance(raw_args, str)
                        else json.dumps(args, ensure_ascii=False, sort_keys=True)
                    ),
                )
                if key in failures:
                    result = {
                        "ok": False,
                        "error": f"原样重试没有意义：刚才同样的调用已失败（{failures[key]}）。"
                        "请修改参数，或先调只读工具核实",
                    }
                elif key in succeeded:
                    result = {
                        "ok": False,
                        "repeated": True,
                        "error": "同样的调用刚才已成功，完整结果就在上文，直接使用即可，"
                        "不要原样重复调用；需要更多数据请换更具体的过滤条件",
                    }
                else:
                    result = execute_tool(ctx, name, args, allow_write)
                    used += 1
                    if result.get("ok", True):
                        succeeded.add(key)
                    else:
                        failures[key] = str(result.get("error", ""))[:120]

            step = {
                "tool": name,
                "args": args,
                "ok": bool(result.get("ok", True)),
                "round": round_no,
                "summary": _summarize(name, result),
            }
            steps.append(step)
            yield {
                "type": "step",
                "phase": "end",
                "tool": name,
                "ok": step["ok"],
                "round": round_no,
                "summary": step["summary"],
                "detail": result,
                "ms": int((time.monotonic() - t_tool) * 1000),
            }
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or name,
                    "content": json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT],
                }
            )

    # 轮次/时长用尽：不许再调工具，强制模型收尾给结论——否则学生只看到一堆工具卡片没答案
    reply = _force_final(convo)
    yield {"type": "state", "state": "finalize"}
    yield {
        "type": "done",
        "reply": reply
        or "（工具调用轮次用尽，先把上面已经写入的结果用起来；"
        "要更细的任务请拆成小问题再问一次。）",
        "steps": steps,
        "rounds": MAX_ROUNDS,
        "tool_calls": used,
        "truncated": True,
        "truncated_by": stop_reason,
        "elapsed_ms": int((time.monotonic() - t_start) * 1000),
    }


def run(
    ctx: AgentContext,
    messages: list,
    allow_write: bool = True,
    on_event: Callable[[dict], None] = None,
) -> dict:
    """非流式包装：跑完返回 done 事件的内容（测试与脚本用）。"""
    final = {"reply": "", "steps": []}
    for ev in run_events(ctx, messages, allow_write):
        if ev["type"] == "done":
            final = ev
        elif on_event:
            on_event(ev)
    return final


def _summarize(name: str, result: dict) -> str:
    """给界面看的一行结果描述。"""
    if result.get("repeated"):
        return "重复调用已拦下（结果上文给过了）"
    if result.get("message"):
        return str(result["message"])[:60]
    if result.get("error"):
        return "失败：" + str(result["error"])[:50]
    if "materials" in result:
        return f"资料 {len(result['materials'])} 篇 / 知识点 {len(result.get('knowledge_points') or [])} 条"
    if "items" in result:
        return f"命中 {result.get('count', len(result['items']))} 条"
    if "text" in result:
        total = result.get("total_chars")
        if total:
            return f"读取 {len(result['text'])} 字（共 {total} 字）"
        return f"读取 {len(result['text'])} 字"
    if "created" in result:
        return f"新增 {result['created']} 题"
    if "matched" in result:
        return f"匹配 {result['matched']} 个章节"
    if "totals" in result:
        t = result["totals"]
        return (
            f"资料 {t.get('materials')} / 知识点 {t.get('kps')} / 错题 {t.get('unresolved_wrong')}"
        )
    return "完成"
