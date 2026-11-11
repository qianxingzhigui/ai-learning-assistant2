"""AI Agent（对话窗口的底座）测试。

不打网络：只测工具层本身——权限边界、写入校验、去重、错误处理。
「模型会不会自己挑工具」属于线上验证（scripts/smoke_agent.py），不进 CI。
"""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ai_agent  # noqa: E402
from app import main as app_main  # noqa: E402


@pytest.fixture
def party_ctx(client):
    """选课后的上下文：uid 来自已登录的随机用户。"""
    client.post("/api/profile/track", json={"track": "party"})
    tree = client.get("/api/syllabus", params={"track": "party"}).json()
    code = tree["chapters"][0]["sections"][0]["code"]
    ctx = ai_agent.AgentContext(client.user_id, "party", app_main.db, app_main.now)
    return ctx, code


def test_工具数量与权限划分():
    read_only = ai_agent.tool_specs(False)
    full = ai_agent.tool_specs(True)
    assert len(read_only) == 11
    assert len(full) == 26
    assert all(
        t["function"]["name"] not in {x["function"]["name"] for x in read_only}
        for t in full
        if t not in read_only
    )
    assert all(not t["write"] for t in ai_agent.TOOLS.values() if t["spec"] in read_only)


def test_只读模式下写入被拒(party_ctx):
    ctx, code = party_ctx
    r = ai_agent.execute_tool(ctx, "create_task", {"title": "只读模式任务"}, allow_write=False)
    assert r["ok"] is False
    assert "写入权限" in r["error"]


def test_建知识点并去重(party_ctx):
    ctx, code = party_ctx
    first = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": code, "title": "测试要点", "description": "描述"},
        allow_write=True,
    )
    assert first["ok"] and first["id"]
    again = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": code, "title": "测试要点", "description": "描述"},
        allow_write=True,
    )
    assert again["duplicated"] is True
    assert again["id"] == first["id"]


def test_拒绝写入别课节点(party_ctx):
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": "prc-01-01", "title": "串课", "description": "x"},
        allow_write=True,
    )
    assert r["ok"] is False
    assert "不属于当前课程" in r["error"]


def test_生成资料入库可检索(party_ctx):
    ctx, code = party_ctx
    body = "遵义会议确立毛泽东在党中央的领导地位，是党的历史上生死攸关的转折点。" * 4
    r = ai_agent.execute_tool(
        ctx,
        "create_material",
        {"title": "测试生成资料", "content": body, "node_code": code},
        allow_write=True,
    )
    assert r["ok"] and r["id"]
    hit = ai_agent.execute_tool(ctx, "search_content", {"query": "遵义会议"}, allow_write=True)
    titles = [m["title"] for m in hit["materials"]]
    assert "测试生成资料" in titles


def test_出题与任务(party_ctx):
    ctx, code = party_ctx
    kp = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": code, "title": "出题用要点", "description": "d"},
        allow_write=True,
    )
    q = ai_agent.execute_tool(
        ctx,
        "create_questions",
        {
            "kp_id": kp["id"],
            "questions": [
                {
                    "type": "单选题",
                    "question": "题干？",
                    "answer": "A",
                    "evidence": "原文句一",
                },
                {
                    "type": "简答",
                    "question": "简述？",
                    "answer": "答案",
                    "evidence": "原文句二",
                },
            ],
        },
        allow_write=True,
    )
    assert q["ok"] and q["created"] == 2

    bad = ai_agent.execute_tool(
        ctx, "create_task", {"title": "坏日期", "due_date": "下周"}, allow_write=True
    )
    assert bad["ok"] is False

    ok = ai_agent.execute_tool(
        ctx,
        "create_task",
        {"title": "复习要点", "due_date": "2030-01-01"},
        allow_write=True,
    )
    assert ok["ok"] and ok["id"]


def test_工具异常不抛给上层(party_ctx):
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(ctx, "read_material", {"material_id": 99999999}, allow_write=True)
    assert r["ok"] is False and "error" in r
    r2 = ai_agent.execute_tool(ctx, "不存在的工具", {}, allow_write=True)
    assert r2["ok"] is False


def test_对话接口无key时给提示(client):
    client.post("/api/profile/track", json={"track": "party"})
    r = client.post(
        "/api/ai/chat",
        json={"messages": [{"role": "user", "content": "你好"}], "stream": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "unavailable"
    assert "API Key" in data["reply"]


def test_未选课时对话接口报错():
    from tests.conftest import AuthClient

    c = AuthClient()  # 新用户 track 为空
    r = c.post("/api/ai/chat", json={"messages": [{"role": "user", "content": "你好"}]})
    assert r.status_code == 400


def test_工具清单接口(client):
    client.post("/api/profile/track", json={"track": "party"})
    data = client.get("/api/ai/tools").json()
    assert len(data["tools"]) == 26
    assert sum(1 for t in data["tools"] if t["write"]) == 15


def _wrong_id(client) -> int:
    """造一条真实错题，返回 wrong_book.id。"""
    client.post("/api/profile/track", json={"track": "party"})
    kp = client.post("/api/knowledge-points", json={"title": "遵义会议", "course": "党史"}).json()
    client.post("/api/questions/generate", json={"kp_id": kp["id"], "num": 1})
    qid = [q for q in client.get("/api/questions").json() if q["kp_id"] == kp["id"]][0]["id"]
    client.post("/api/practice/submit", json={"question_id": qid, "correct": False})
    return client.get("/api/wrong-book").json()[0]["id"]


def test_错因归因打标签(client):
    wid = _wrong_id(client)
    ctx = ai_agent.AgentContext(client.user_id, "party", app_main.db, app_main.now)
    r = ai_agent.execute_tool(
        ctx,
        "tag_wrong_error",
        {"wrong_id": wid, "error_type": "sequence", "advice": "先把时间轴串一遍"},
        allow_write=True,
    )
    assert r["ok"] is True and r["error_type"] == "时序错位"
    item = client.get("/api/wrong-book").json()[0]
    assert item["error_type"] == "时序错位"
    # 也接受直接写中文分类
    r2 = ai_agent.execute_tool(
        ctx,
        "tag_wrong_error",
        {"wrong_id": wid, "error_type": "记忆不牢"},
        allow_write=True,
    )
    assert r2["ok"] is True
    assert client.get("/api/wrong-book").json()[0]["error_type"] == "记忆不牢"


def test_错因归因的边界(client):
    wid = _wrong_id(client)
    ctx = ai_agent.AgentContext(client.user_id, "party", app_main.db, app_main.now)
    # 分类不在词表里 → 拒绝，提示可选值
    bad = ai_agent.execute_tool(
        ctx,
        "tag_wrong_error",
        {"wrong_id": wid, "error_type": "心情不好"},
        allow_write=True,
    )
    assert bad["ok"] is False and "error_type" in bad["error"]
    # 别人的错题 → 拒绝（用户隔离）
    other = ai_agent.AgentContext(client.user_id + 999999, "party", app_main.db, app_main.now)
    mine = ai_agent.execute_tool(
        other,
        "tag_wrong_error",
        {"wrong_id": wid, "error_type": "concept"},
        allow_write=True,
    )
    assert mine["ok"] is False
    # 只读模式 → 拒绝
    ro = ai_agent.execute_tool(
        ctx,
        "tag_wrong_error",
        {"wrong_id": wid, "error_type": "concept"},
        allow_write=False,
    )
    assert ro["ok"] is False and "写入权限" in ro["error"]


def test_记忆沉淀与查重(party_ctx):
    ctx, _ = party_ctx
    r1 = ai_agent.execute_tool(
        ctx,
        "save_memory",
        {"kind": "weakness", "content": "时序题反复错"},
        allow_write=True,
    )
    assert r1["ok"] and "duplicated" not in r1
    r2 = ai_agent.execute_tool(
        ctx,
        "save_memory",
        {"kind": "weakness", "content": "时序题反复错"},
        allow_write=True,
    )
    assert r2["duplicated"] is True
    # 非法 kind 拒绝
    bad = ai_agent.execute_tool(
        ctx, "save_memory", {"kind": "other", "content": "x"}, allow_write=True
    )
    assert bad["ok"] is False
    items = ai_agent.execute_tool(ctx, "list_memory", {}, allow_write=True)
    assert items["ok"] and any("时序题反复错" in i["content"] for i in items["items"])


def test_记忆只读模式不可写(party_ctx):
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(
        ctx, "save_memory", {"kind": "profile", "content": "大三"}, allow_write=False
    )
    assert r["ok"] is False and "写入权限" in r["error"]


# ---------------- 状态机（act → review → 回流 → finalize） ----------------
# 不打网络：把 chat_completion 换成脚本化的假模型，把质检员换成预设裁决，
# 只验状态迁移本身是否按预期发生。


def _scripted_chat(script):
    """script: [(kind, payload)]，kind = 'tools' 或 'text'。返回 (fake, calls)。"""
    calls = []

    def fake(convo, tools=None, **kw):
        calls.append([dict(m) for m in convo])
        kind, payload = script[min(len(calls) - 1, len(script) - 1)]
        if kind == "tools":
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{len(calls)}",
                        "type": "function",
                        "function": {"name": payload, "arguments": "{}"},
                    }
                ],
            }
        return {"content": payload, "tool_calls": []}

    return fake, calls


def _states(events):
    return [e.get("state") for e in events if e.get("type") == "state"]


def test_状态机_审查不达标会回流(party_ctx, monkeypatch):
    """质检员第一次判不达标 → 必须回到 act 并把缺口喂回模型，再 review 才收尾。"""
    ctx, _ = party_ctx
    fake, calls = _scripted_chat(
        [
            ("tools", "get_my_stats"),
            ("text", "第一稿：你最近在学党史。"),
            ("text", "第二稿：已按你的情况建好 3 天复习任务。"),
        ]
    )
    verdicts = [
        {"done": False, "gap": "学生要的是复习计划，你只报了个进度"},
        {"done": True, "gap": ""},
    ]
    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)
    monkeypatch.setattr(ai_agent, "_review_draft", lambda goal, steps, draft: verdicts.pop(0))

    events = list(
        ai_agent.run_events(
            ctx, [{"role": "user", "content": "按我情况排个复习计划"}], allow_write=True
        )
    )

    assert _states(events) == ["act", "review", "act", "review", "finalize"]
    # 回流时必须把审查意见作为新一轮输入喂回去
    reflow_msg = calls[2][-1]
    assert reflow_msg["role"] == "user" and "质量审查" in reflow_msg["content"]
    assert "复习计划" in reflow_msg["content"]
    done = events[-1]
    assert done["type"] == "done" and "复习任务" in done["reply"]
    assert done["reviews"] == 2 and done["tool_calls"] == 1


def test_轨迹事件_含思考参数与耗时(party_ctx, monkeypatch):
    """前端「执行轨迹」要靠这些事件：模型的中间想法、每次调用的参数与耗时。"""
    ctx, _ = party_ctx

    def fake(convo, tools=None, **kw):
        if not any(m.get("role") == "tool" for m in convo):
            return {
                "content": "先看看你最近的学习数据",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_my_stats", "arguments": "{}"},
                    }
                ],
            }
        return {"content": "你目前的情况是……", "tool_calls": []}

    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)
    monkeypatch.setattr(ai_agent, "_review_draft", lambda *a: {"done": True, "gap": ""})
    events = list(
        ai_agent.run_events(
            ctx, [{"role": "user", "content": "我最近学得怎么样"}], allow_write=True
        )
    )

    thinks = [e for e in events if e.get("type") == "thinking"]
    assert thinks and thinks[0]["round"] == 1 and "学习数据" in thinks[0]["text"]
    starts = [e for e in events if e.get("type") == "step" and e.get("phase") == "start"]
    ends = [e for e in events if e.get("type") == "step" and e.get("phase") == "end"]
    assert starts[0]["tool"] == "get_my_stats" and starts[0]["round"] == 1
    assert starts[0]["args"] == {}  # 参数原样下发，界面能显示 AI 到底传了什么
    assert isinstance(ends[0]["ms"], int) and ends[0]["ms"] >= 0
    assert events[-1]["elapsed_ms"] >= 0


def test_状态机_回流有上限(party_ctx, monkeypatch):
    """质检员一直不达标也不能死循环：到 MAX_REVIEWS 就收尾。"""
    ctx, _ = party_ctx
    fake, _calls = _scripted_chat([("tools", "get_my_stats"), ("text", "总是差一点的答复")])
    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)
    monkeypatch.setattr(
        ai_agent,
        "_review_draft",
        lambda goal, steps, draft: {"done": False, "gap": "还是不行"},
    )

    events = list(
        ai_agent.run_events(ctx, [{"role": "user", "content": "随便看看"}], allow_write=True)
    )

    assert _states(events).count("review") == ai_agent.MAX_REVIEWS
    assert _states(events)[-1] == "finalize"
    assert events[-1]["type"] == "done"


def test_状态机_纯问答跳过审查(party_ctx, monkeypatch):
    """没动过工具 = 纯问答，不该白烧一次质检调用。"""
    ctx, _ = party_ctx
    fake, _calls = _scripted_chat([("text", "遵义会议是 1935 年 1 月召开的。")])
    reviewed = []
    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)
    monkeypatch.setattr(
        ai_agent,
        "_review_draft",
        lambda *a: reviewed.append(1) or {"done": True, "gap": ""},
    )

    events = list(
        ai_agent.run_events(
            ctx, [{"role": "user", "content": "遵义会议哪年开的"}], allow_write=True
        )
    )

    assert _states(events) == ["act", "finalize"]
    assert reviewed == []
    assert events[-1]["type"] == "done" and "1935" in events[-1]["reply"]


# ---------------- 防审查死循环（2026-11-09 真机 HTTP 451 事故） ----------------


def test_状态机_重复调用被短路(party_ctx, monkeypatch):
    """同样的工具+参数刚成功过就该短路提示，别再把大结果原文灌一遍上下文。"""
    ctx, _ = party_ctx
    fake, calls = _scripted_chat(
        [
            ("tools", "get_my_stats"),
            ("tools", "get_my_stats"),
            ("text", "已看完你的学习数据。"),
        ]
    )
    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)
    monkeypatch.setattr(ai_agent, "_review_draft", lambda *a: {"done": True, "gap": ""})

    events = list(ai_agent.run_events(ctx, [{"role": "user", "content": "看看我的情况"}]))

    done = events[-1]
    assert done["type"] == "done" and done["tool_calls"] == 1  # 第二次没真正执行
    # 第二次的 tool 回灌是短路提示，不是完整结果
    tool_msgs = [m for m in calls[2] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "不要原样重复调用" in tool_msgs[1]["content"]
    # 界面摘要要能说明是被拦下的重复调用
    dup_steps = [e for e in events if e.get("type") == "step" and e.get("phase") == "end"]
    assert "重复调用" in dup_steps[-1]["summary"]


def test_状态机_审查拦截摘原文重试(party_ctx, monkeypatch):
    """HTTP 451 censorship：带大段工具原文被拦时，摘掉原文救一次而不是直接报错。"""
    ctx, _ = party_ctx
    calls = []

    def fake(convo, tools=None, **kw):
        calls.append([dict(m) for m in convo])
        if len(calls) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_knowledge_points",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        if len(calls) == 2:
            raise RuntimeError('HTTP 451 {"error": {"type": "censorship_blocked"}}')
        return {"content": "好的，基于前面查到的知识点给你结论。", "tool_calls": []}

    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)
    monkeypatch.setattr(ai_agent, "_review_draft", lambda *a: {"done": True, "gap": ""})

    events = list(ai_agent.run_events(ctx, [{"role": "user", "content": "把知识点过一遍"}]))

    # 不出 error 事件，救回来正常收尾
    assert not [e for e in events if e["type"] == "error"]
    assert events[-1]["type"] == "done" and events[-1]["reply"]
    # 第三次调用时 tool 原文已被摘掉
    tool_msgs = [m for m in calls[2] if m.get("role") == "tool"]
    assert tool_msgs and "因平台内容审查被省略" in tool_msgs[0]["content"]


def test_审查拦截直接报错是人话(party_ctx, monkeypatch):
    """没有工具原文可摘时（纯问答被拦），报错要能看懂而不是甩一串 HTTP 451。"""
    ctx, _ = party_ctx

    def fake(convo, tools=None, **kw):
        raise RuntimeError('HTTP 451 {"error": {"type": "censorship_blocked"}}')

    monkeypatch.setattr(ai_agent.ai_helper, "chat_completion", fake)

    events = list(ai_agent.run_events(ctx, [{"role": "user", "content": "随便聊聊"}]))

    errs = [e for e in events if e["type"] == "error"]
    assert errs and "安全审查" in errs[0]["message"] and "451" not in errs[0]["message"]


def test_list知识点_上限30(party_ctx):
    """limit 传 60 也最多给 30 条：大结果反复回灌是审查拦截的根源。"""
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(ctx, "list_knowledge_points", {"limit": 60}, allow_write=False)
    assert "error" not in r
    assert len(r["items"]) <= 30


# ---------------- 删除类工具（2026-11-09 开放） ----------------


def test_删除任务(party_ctx):
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(ctx, "create_task", {"title": "待删任务"}, allow_write=True)
    tid = r["id"]
    d = ai_agent.execute_tool(ctx, "delete_task", {"id": tid}, allow_write=True)
    assert d["ok"] and d["deleted"] == tid
    # 再删一次应报不存在，而不是静默成功
    d2 = ai_agent.execute_tool(ctx, "delete_task", {"id": tid}, allow_write=True)
    assert d2["ok"] is False and "不存在" in d2["error"]


def test_删除必带id且不许跨用户(party_ctx):
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(ctx, "delete_task", {}, allow_write=True)
    assert r["ok"] is False and "id" in r["error"]
    # 别人的任务删不到（user_id 隔离）
    r2 = ai_agent.execute_tool(ctx, "delete_task", {"id": 999999}, allow_write=True)
    assert r2["ok"] is False and "不存在" in r2["error"]


def test_删除题目级联清错题(party_ctx):
    ctx, code = party_ctx
    kp = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": code, "title": "级联测试点"},
        allow_write=True,
    )
    qs = ai_agent.execute_tool(
        ctx,
        "create_questions",
        {
            "kp_id": kp["id"],
            "questions": [
                {
                    "type": "简答",
                    "question": "级联测试题？",
                    "answer": "答案",
                    "evidence": "原文依据",
                }
            ],
        },
        allow_write=True,
    )
    qid = qs["ids"][0]
    # 制造错题记录
    ctx_conn = ctx.conn()
    ctx_conn.execute(
        "INSERT INTO wrong_book(user_id,question_id,wrong_count,resolved,uuid,last_wrong_at)"
        " VALUES(?,?,1,0,?,?)",
        (ctx.uid, qid, uuid.uuid4().hex, ctx.now()),
    )
    ctx_conn.commit()
    ctx_conn.close()
    d = ai_agent.execute_tool(ctx, "delete_question", {"id": qid}, allow_write=True)
    assert d["ok"]
    conn = ctx.conn()
    left_q = conn.execute("SELECT COUNT(*) FROM questions WHERE id=?", (qid,)).fetchone()[0]
    left_w = conn.execute("SELECT COUNT(*) FROM wrong_book WHERE question_id=?", (qid,)).fetchone()[
        0
    ]
    conn.close()
    assert left_q == 0 and left_w == 0


def test_清空错题本(party_ctx):
    ctx, code = party_ctx
    kp = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": code, "title": "清空测试点"},
        allow_write=True,
    )
    qs = ai_agent.execute_tool(
        ctx,
        "create_questions",
        {
            "kp_id": kp["id"],
            "questions": [
                {
                    "type": "简答",
                    "question": "清空测试题？",
                    "answer": "答案",
                    "evidence": "原文依据",
                }
            ],
        },
        allow_write=True,
    )
    conn = ctx.conn()
    conn.execute(
        "INSERT INTO wrong_book(user_id,question_id,wrong_count,resolved,uuid,last_wrong_at)"
        " VALUES(?,?,1,0,?,?)",
        (ctx.uid, qs["ids"][0], uuid.uuid4().hex, ctx.now()),
    )
    conn.commit()
    conn.close()
    r = ai_agent.execute_tool(ctx, "clear_wrong_book", {}, allow_write=True)
    assert r["ok"] and r["cleared"] >= 1
    conn = ctx.conn()
    n = conn.execute("SELECT COUNT(*) FROM wrong_book WHERE user_id=?", (ctx.uid,)).fetchone()[0]
    conn.close()
    assert n == 0


def test_删除在只读模式下被拒(party_ctx):
    ctx, _ = party_ctx
    r = ai_agent.execute_tool(ctx, "clear_wrong_book", {}, allow_write=False)
    assert r["ok"] is False and "写入权限" in r["error"]
    r2 = ai_agent.execute_tool(ctx, "delete_task", {"id": 1}, allow_write=False)
    assert r2["ok"] is False and "写入权限" in r2["error"]


def test_单选题选项拼题干与答案归一(party_ctx):
    ctx, code = party_ctx
    kp = ai_agent.execute_tool(
        ctx,
        "create_knowledge_point",
        {"node_code": code, "title": "选择题测试点"},
        allow_write=True,
    )
    # 结构化 options：拼进题干，答案字母归一
    r1 = ai_agent.execute_tool(
        ctx,
        "create_questions",
        {
            "kp_id": kp["id"],
            "questions": [
                {
                    "type": "单选题",
                    "question": "党的十一届三中全会召开于哪一年？",
                    "options": ["1976年", "1978年", "1980年", "1982年"],
                    "answer": "B",
                    "evidence": "1978年12月，党的十一届三中全会召开。",
                },
                {
                    "type": "单选题",
                    "question": "拨乱反正胜利完成的标志是？",
                    "options": ["十一届三中全会", "历史决议通过", "真理标准讨论"],
                    "answer": "历史决议通过",
                    "evidence": "历史决议的通过标志着拨乱反正胜利完成。",
                },
            ],
        },
        allow_write=True,
    )
    assert r1["ok"] and len(r1["ids"]) == 2
    conn = ctx.conn()
    rows = conn.execute(
        "SELECT type,question,answer FROM questions WHERE id IN (?,?)", r1["ids"]
    ).fetchall()
    conn.close()
    q1, q2 = rows[0], rows[1]
    assert q1["type"] == "单选题" and q1["answer"].upper().startswith("B")
    assert "A. 1976年" in q1["question"] and "D. 1982年" in q1["question"]
    # 选项全文给的答案也归一成字母
    assert q2["answer"].upper().startswith("B"), q2["answer"]
    # 答案已带字母的保持原样
    r2 = ai_agent.execute_tool(
        ctx,
        "create_questions",
        {
            "kp_id": kp["id"],
            "questions": [
                {
                    "type": "单选题",
                    "question": "题干已含选项：？A. 甲 B. 乙 C. 丙 D. 丁",
                    "answer": "C. 丙",
                    "evidence": "依据",
                }
            ],
        },
        allow_write=True,
    )
    conn = ctx.conn()
    row = conn.execute(
        "SELECT question,answer FROM questions WHERE id=?", (r2["ids"][0],)
    ).fetchone()
    conn.close()
    assert row["question"].count("A.") == 1  # 没有重复拼选项
    assert row["answer"].upper().startswith("C")
    # 模型把选项改写一遍再当答案交（2026-11-10 实测）——模糊兜底也要换算成字母
    r3 = ai_agent.execute_tool(
        ctx,
        "create_questions",
        {
            "kp_id": kp["id"],
            "questions": [
                {
                    "type": "单选题",
                    "question": "战略判断是？\nA. 战争与革命是时代主题\nB. 和平与发展是时代主题\nC. 国防建设优先于经济建设\nD. 早打大打打核战争",
                    "answer": "和平与发展是当代世界的两大问题",  # 与 B 选项文字不重合的改写
                    "evidence": "邓小平作出和平与发展是时代主题的判断。",
                },
                {
                    "type": "单选题",
                    "question": "完全对不上选项的？\nA. 甲乙丙\nB. 丁戊己\nC. 庚辛壬",
                    "answer": "八竿子打不着的答案",  # 模糊兜底失败 → 原样入库，前端回退自评
                    "evidence": "依据",
                },
            ],
        },
        allow_write=True,
    )
    assert r3["ok"] and len(r3["ids"]) == 2
    conn = ctx.conn()
    rows = conn.execute(
        "SELECT answer FROM questions WHERE id IN (?,?) ORDER BY id", r3["ids"]
    ).fetchall()
    conn.close()
    assert rows[0][0].startswith("B"), rows[0][0]  # 改写答案 → 换算成字母
    assert rows[1][0] == "八竿子打不着的答案"  # 对不上就原样返回，前端自评兜底
