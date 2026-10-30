# -*- coding: utf-8 -*-
"""跨机同步核心：uuid 键控的导出 / 合并。

设计要点
- 每行以 uuid 为唯一身份，本地自增 id 只在本机有意义。
- 外键在 payload 里一律用 uuid 表达（material_uuid / kp_uuid / question_uuid / owner）。
- 合并只插本地没有的 uuid，绝不覆盖已有行 → 双向同步天然无冲突（追加型数据）。
- 同名用户并入现有账号（demo 在两台机器上 uuid 不同，但视为同一账号）。
- 纯函数：conn 由调用方（main.py 路由）传入，避免循环导入。
"""

import sqlite3
import uuid

# 表顺序 = 外键依赖顺序，导入必须按它来
CHAIN = (
    "users",
    "materials",
    "knowledge_points",
    "questions",
    "practice_records",
    "wrong_book",
    "tasks",
)

PAYLOAD_VERSION = 1


def _uuid(row) -> str:
    return row["uuid"] or ""


# ==================== 导出 ====================
def build_export(conn: sqlite3.Connection) -> dict:
    """全量导出当前库（数据量级：个人学习记录，几十 KB，全量即可不用增量）。"""
    users = conn.execute(
        "SELECT uuid, username, password_hash, salt, display_name, created_at FROM users"
    ).fetchall()

    def rows(sql):
        return [dict(r) for r in conn.execute(sql).fetchall()]

    payload = {
        "version": PAYLOAD_VERSION,
        "users": [dict(u) for u in users],
        "materials": rows("""
            SELECT m.uuid, u.uuid AS owner, m.title, m.course, m.filename, m.filetype,
                   m.content, m.created_at
            FROM materials m JOIN users u ON u.id = m.user_id"""),
        "knowledge_points": rows("""
            SELECT k.uuid, u.uuid AS owner, m.uuid AS material_uuid,
                   k.course, k.title, k.description, k.created_at
            FROM knowledge_points k
            JOIN users u ON u.id = k.user_id
            LEFT JOIN materials m ON m.id = k.material_id"""),
        "questions": rows("""
            SELECT q.uuid, u.uuid AS owner, k.uuid AS kp_uuid,
                   q.type, q.question, q.answer, q.source, q.created_at
            FROM questions q
            JOIN users u ON u.id = q.user_id
            LEFT JOIN knowledge_points k ON k.id = q.kp_id"""),
        "practice_records": rows("""
            SELECT p.uuid, u.uuid AS owner, q.uuid AS question_uuid,
                   p.user_answer, p.correct, p.created_at
            FROM practice_records p
            JOIN users u ON u.id = p.user_id
            JOIN questions q ON q.id = p.question_id"""),
        "wrong_book": rows("""
            SELECT w.uuid, u.uuid AS owner, q.uuid AS question_uuid,
                   w.wrong_count, w.resolved, w.review_marked, w.last_wrong_at
            FROM wrong_book w
            JOIN users u ON u.id = w.user_id
            JOIN questions q ON q.id = w.question_id"""),
        "tasks": rows("""
            SELECT t.uuid, u.uuid AS owner, k.uuid AS kp_uuid,
                   t.course, t.title, t.description, t.due_date, t.status, t.created_at
            FROM tasks t
            JOIN users u ON u.id = t.user_id
            LEFT JOIN knowledge_points k ON k.id = t.kp_id"""),
    }
    payload["counts"] = {k: len(v) for k, v in payload.items() if isinstance(v, list)}
    return payload


# ==================== 合并导入 ====================
def apply_import(conn: sqlite3.Connection, payload: dict) -> dict:
    """按 uuid 合并：只插本地缺失的行，返回插入统计。

    返回 {table: inserted}，外键通过 uuid→本地 id 的映射表重接。
    """
    stats = {t: 0 for t in CHAIN if t in payload}
    uid_map = {}  # 远端 user uuid -> 本地 users.id
    mat_map = {}  # 远端 material uuid -> 本地 materials.id
    kp_map = {}  # 远端 kp uuid -> 本地 knowledge_points.id
    q_map = {}  # 远端 question uuid -> 本地 questions.id

    def have_uuid(table):
        return {r[0] for r in conn.execute(f"SELECT uuid FROM {table} WHERE uuid IS NOT NULL")}

    def uuid_exists(table, u):
        return conn.execute(f"SELECT 1 FROM {table} WHERE uuid=?", (u,)).fetchone() is not None

    def resolve_user(owner_uuid):
        """远端 owner uuid -> 本地 user id；同名并入现有账号。"""
        if not owner_uuid:
            return 1
        if owner_uuid in uid_map:
            return uid_map[owner_uuid]
        row = conn.execute("SELECT id FROM users WHERE uuid=?", (owner_uuid,)).fetchone()
        if row:
            uid_map[owner_uuid] = row["id"]
            return row["id"]
        src = next((u for u in payload.get("users", []) if u.get("uuid") == owner_uuid), None)
        if src:
            same_name = conn.execute(
                "SELECT id FROM users WHERE username=?", (src["username"],)
            ).fetchone()
            if same_name:  # 同名：视为同一人（demo 两台机器 uuid 不同就是这种情况）
                uid_map[owner_uuid] = same_name["id"]
                return same_name["id"]
            cur = conn.execute(
                "INSERT INTO users(uuid,username,password_hash,salt,display_name,created_at) VALUES(?,?,?,?,?,?)",
                (
                    owner_uuid,
                    src["username"],
                    src["password_hash"],
                    src["salt"],
                    src.get("display_name", ""),
                    src.get("created_at", ""),
                ),
            )
            uid_map[owner_uuid] = cur.lastrowid
            stats["users"] += 1
            return cur.lastrowid
        return 1  # 孤儿数据兜底：挂到 demo

    def null_fk(map_, fk_uuid):
        if not fk_uuid:
            return None
        if fk_uuid in map_:
            return map_[fk_uuid]
        return None  # 引用主体没同步过来就置空，行本身不丢

    conn.execute("PRAGMA foreign_keys=OFF")  # 批量导入期间手工保证引用顺序

    # 1. users（未在 resolve_user 里建过的孤立账号也收进来）
    for u in payload.get("users", []):
        u_uuid = u.get("uuid")
        if not u_uuid or uuid_exists("users", u_uuid):
            continue
        if conn.execute("SELECT 1 FROM users WHERE username=?", (u["username"],)).fetchone():
            continue  # 同名并入，不建新号
        conn.execute(
            "INSERT INTO users(uuid,username,password_hash,salt,display_name,created_at) VALUES(?,?,?,?,?,?)",
            (
                u_uuid,
                u["username"],
                u["password_hash"],
                u["salt"],
                u.get("display_name", ""),
                u.get("created_at", ""),
            ),
        )
        stats["users"] += 1

    # 2. materials
    for r in payload.get("materials", []):
        if not r.get("uuid") or uuid_exists("materials", r["uuid"]):
            continue
        conn.execute(
            "INSERT INTO materials(user_id,uuid,title,course,filename,filetype,content,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                resolve_user(r["owner"]),
                r["uuid"],
                r["title"],
                r["course"],
                r.get("filename", ""),
                r.get("filetype", "txt"),
                r.get("content", ""),
                r.get("created_at", ""),
            ),
        )
        stats["materials"] += 1
    for r in conn.execute("SELECT id, uuid FROM materials WHERE uuid IS NOT NULL"):
        mat_map[r["uuid"]] = r["id"]

    # 3. knowledge_points
    for r in payload.get("knowledge_points", []):
        if not r.get("uuid") or uuid_exists("knowledge_points", r["uuid"]):
            continue
        conn.execute(
            "INSERT INTO knowledge_points(user_id,uuid,material_id,course,title,description,created_at) VALUES(?,?,?,?,?,?,?)",
            (
                resolve_user(r["owner"]),
                r["uuid"],
                null_fk(mat_map, r.get("material_uuid")),
                r.get("course", "未分类"),
                r["title"],
                r.get("description", ""),
                r.get("created_at", ""),
            ),
        )
        stats["knowledge_points"] += 1
    for r in conn.execute("SELECT id, uuid FROM knowledge_points WHERE uuid IS NOT NULL"):
        kp_map[r["uuid"]] = r["id"]

    # 4. questions
    for r in payload.get("questions", []):
        if not r.get("uuid") or uuid_exists("questions", r["uuid"]):
            continue
        conn.execute(
            "INSERT INTO questions(user_id,uuid,kp_id,type,question,answer,source,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                resolve_user(r["owner"]),
                r["uuid"],
                null_fk(kp_map, r.get("kp_uuid")),
                r.get("type", "short_answer"),
                r["question"],
                r.get("answer", ""),
                r.get("source", "manual"),
                r.get("created_at", ""),
            ),
        )
        stats["questions"] += 1
    for r in conn.execute("SELECT id, uuid FROM questions WHERE uuid IS NOT NULL"):
        q_map[r["uuid"]] = r["id"]

    # 5. practice_records（题目没同步过来就跳过，否则悬空）
    for r in payload.get("practice_records", []):
        if not r.get("uuid") or uuid_exists("practice_records", r["uuid"]):
            continue
        qid = q_map.get(r.get("question_uuid"))
        if not qid:
            continue
        conn.execute(
            "INSERT INTO practice_records(user_id,uuid,question_id,user_answer,correct,created_at) VALUES(?,?,?,?,?,?)",
            (
                resolve_user(r["owner"]),
                r["uuid"],
                qid,
                r.get("user_answer", ""),
                1 if r.get("correct") else 0,
                r.get("created_at", ""),
            ),
        )
        stats["practice_records"] += 1

    # 6. wrong_book（同题已有错题记录则累加次数，不另建行）
    for r in payload.get("wrong_book", []):
        if not r.get("uuid") or uuid_exists("wrong_book", r["uuid"]):
            continue
        qid = q_map.get(r.get("question_uuid"))
        if not qid:
            continue
        owner = resolve_user(r["owner"])
        exist = conn.execute(
            "SELECT id, wrong_count FROM wrong_book WHERE question_id=? AND user_id=?", (qid, owner)
        ).fetchone()
        if exist:
            conn.execute(
                "UPDATE wrong_book SET wrong_count=wrong_count+? WHERE id=?",
                (int(r.get("wrong_count", 1)), exist["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO wrong_book(user_id,uuid,question_id,wrong_count,resolved,review_marked,last_wrong_at) VALUES(?,?,?,?,?,?,?)",
                (
                    owner,
                    r["uuid"],
                    qid,
                    int(r.get("wrong_count", 1)),
                    1 if r.get("resolved") else 0,
                    1 if r.get("review_marked") else 0,
                    r.get("last_wrong_at", ""),
                ),
            )
            stats["wrong_book"] += 1

    # 7. tasks
    for r in payload.get("tasks", []):
        if not r.get("uuid") or uuid_exists("tasks", r["uuid"]):
            continue
        conn.execute(
            "INSERT INTO tasks(user_id,uuid,course,kp_id,title,description,due_date,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                resolve_user(r["owner"]),
                r["uuid"],
                r.get("course", "未分类"),
                null_fk(kp_map, r.get("kp_uuid")),
                r["title"],
                r.get("description", ""),
                r.get("due_date"),
                r.get("status", "todo"),
                r.get("created_at", ""),
            ),
        )
        stats["tasks"] += 1

    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    return stats
