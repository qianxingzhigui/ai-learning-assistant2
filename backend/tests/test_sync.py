# -*- coding: utf-8 -*-
"""同步核心测试：uuid 迁移 + 导出/合并往返。

不碰网络：push/pull 的 GitHub 传输层在端到端阶段手工验证，
这里专注验证「导出的 payload 能在另一个库无损合并」这一核心正确性。
"""

import uuid as uuid_mod

from app import main as app_main
from app.sync import apply_import, build_export
from fastapi.testclient import TestClient
from tests.conftest import AuthClient, attach_auth


def _make_content(client: AuthClient) -> None:
    """给客户端账号造一套完整链路数据：资料→知识点→题目→练习→错题→任务。"""
    # 手工插资料（走 API 要传文件，直接落库更快更可控）
    conn = app_main.db()
    conn.execute(
        "INSERT INTO materials(user_id,uuid,title,course,filename,filetype,content,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (
            client.user_id,
            uuid_mod.uuid4().hex,
            "测试资料",
            "测试课",
            "t.txt",
            "txt",
            "监督学习是机器学习的一种方法。" * 5,
            "2026-10-28T09:00:00",
        ),
    )
    conn.commit()
    conn.close()
    r = client.post("/api/knowledge-points", json={"title": "监督学习", "course": "测试课"})
    assert r.status_code == 200
    kid = r.json()["id"]
    r = client.post("/api/tasks", json={"title": "复习监督学习", "course": "测试课", "kp_id": kid})
    assert r.status_code == 200
    # 题目 + 练习 + 错题
    conn = app_main.db()
    cur = conn.execute(
        "INSERT INTO questions(user_id,uuid,kp_id,type,question,answer,source,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (
            client.user_id,
            uuid_mod.uuid4().hex,
            kid,
            "short_answer",
            "监督学习需要什么？",
            "带标签的数据",
            "manual",
            "2026-10-28T09:00:00",
        ),
    )
    qid = cur.lastrowid
    conn.commit()
    conn.close()
    r = client.post(
        "/api/practice/submit", json={"question_id": qid, "correct": False, "answer": "不知道"}
    )
    assert r.status_code == 200
    # 再补一次答对
    r = client.post(
        "/api/practice/submit", json={"question_id": qid, "correct": True, "answer": "带标签的数据"}
    )
    assert r.status_code == 200


def test_uuid_migration_backfills_all_rows(tmp_path, monkeypatch):
    """老库（无 uuid 列）启动后所有行都有 uuid 且唯一。"""
    db_path = tmp_path / "legacy.db"
    import sqlite3

    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT,
                           salt TEXT, display_name TEXT DEFAULT '', created_at TEXT);
        INSERT INTO users VALUES(1,'demo','h','s','演示','2026-01-01');
        CREATE TABLE materials(id INTEGER PRIMARY KEY, user_id INTEGER DEFAULT 1,
                               title TEXT, course TEXT DEFAULT '', filename TEXT,
                               filetype TEXT, content TEXT, created_at TEXT);
        INSERT INTO materials VALUES(1,1,'老资料','c','f','txt','内容','2026-01-01');
    """)
    con.commit()
    con.close()

    monkeypatch.setattr(app_main.apppaths, "app_base_dir", lambda: tmp_path)
    app_main.DB_PATH = db_path
    app_main.UPLOAD_DIR = tmp_path / "data" / "uploads"
    app_main.DATA_DIR = tmp_path / "data"
    app_main.init_db()

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    for t in ("users", "materials"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        assert "uuid" in cols, f"{t} 缺 uuid 列"
        nulls = con.execute(f"SELECT COUNT(*) FROM {t} WHERE uuid IS NULL OR uuid=''").fetchone()[0]
        assert nulls == 0, f"{t} 有 {nulls} 行没回填 uuid"
    assert con.execute("SELECT uuid FROM users WHERE id=1").fetchone()[0]
    con.close()


def test_export_import_roundtrip(tmp_path, monkeypatch):
    """甲机的全部数据合并进乙机空库后：行数一致、外键重接正确、账号可登录。"""
    # 甲机：造数据
    db_a = tmp_path / "a"
    monkeypatch.setattr(app_main.apppaths, "app_base_dir", lambda: db_a)
    app_main.DB_PATH = db_a / "data" / "app.db"
    app_main.UPLOAD_DIR = db_a / "data" / "uploads"
    app_main.DATA_DIR = db_a / "data"
    app_main.SYNC_CONFIG_PATH = app_main.DATA_DIR / "sync.json"
    app_main.init_db()
    a = AuthClient()
    _make_content(a)
    conn = app_main.db()
    payload = build_export(conn)
    conn.close()
    assert payload["counts"]["materials"] == 1
    assert payload["counts"]["practice_records"] == 2
    assert payload["counts"]["wrong_book"] == 1
    assert payload["counts"]["tasks"] == 1

    # 乙机：全新的空库（只有它自己的账号）
    db_b = tmp_path / "b"
    monkeypatch.setattr(app_main.apppaths, "app_base_dir", lambda: db_b)
    app_main.DB_PATH = db_b / "data" / "app.db"
    app_main.UPLOAD_DIR = db_b / "data" / "uploads"
    app_main.DATA_DIR = db_b / "data"
    app_main.SYNC_CONFIG_PATH = app_main.DATA_DIR / "sync.json"
    app_main.init_db()
    b = AuthClient()  # 乙机本地已有一个自己的账号
    conn = app_main.db()
    stats = apply_import(conn, payload)
    conn.close()

    # 甲的账号被原样搬过来（uuid 相同），数据行数一致
    src_user = next(u for u in payload["users"] if u["username"] == a.username)
    conn = app_main.db()
    imported_user = conn.execute("SELECT * FROM users WHERE uuid=?", (src_user["uuid"],)).fetchone()
    assert imported_user is not None, "甲的账号没合并进来"
    assert imported_user["username"] == a.username

    # 锁死结果，再统计行数，避免中途被别的连接改写
    n_materials = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
    n_practice = conn.execute("SELECT COUNT(*) FROM practice_records").fetchone()[0]
    n_wrong = conn.execute("SELECT COUNT(*) FROM wrong_book").fetchone()[0]
    n_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    # 乙库原本为空（仅 1 个自带账号），导入甲后应正好等于甲的数量
    assert n_materials == 1, f"materials 应为 1（仅甲带过来的那条），实际 {n_materials}"
    assert n_practice == 2, f"practice_records 应为 2，实际 {n_practice}"
    assert n_wrong == 1, f"wrong_book 应为 1，实际 {n_wrong}"
    assert n_tasks == 1, f"tasks 应为 1，实际 {n_tasks}"
    # 外键重接：练习记录指向本地刚导入的题目
    fk_ok = conn.execute(
        """
        SELECT COUNT(*) FROM practice_records p
        JOIN questions q ON q.id = p.question_id
        JOIN users u ON u.id = p.user_id
        WHERE u.username = ?
    """,
        (a.username,),
    ).fetchone()[0]
    assert fk_ok == 2, "练习记录外键没有重接到导入的题目/用户"
    conn.close()

    # 合并后的账号能用原密码登录（password_hash 原样搬过来了）
    c = TestClient(app_main.app)
    r = c.post("/api/auth/login", json={"username": a.username, "password": "test123456"})
    assert r.status_code == 200, "导入账号无法登录"

    # 幂等：同一份 payload 再导一次，行数不变
    conn = app_main.db()
    stats2 = apply_import(conn, payload)
    conn.close()
    assert all(v == 0 for v in stats2.values()), f"重复导入产生了重复行: {stats2}"


def test_sync_routes_contract(tmp_path, monkeypatch):
    """export/import 路由可用；push 未配置时给出友好 400。"""
    monkeypatch.setattr(app_main.apppaths, "app_base_dir", lambda: tmp_path)
    app_main.DB_PATH = tmp_path / "data" / "app.db"
    app_main.UPLOAD_DIR = tmp_path / "data" / "uploads"
    app_main.DATA_DIR = tmp_path / "data"
    app_main.SYNC_CONFIG_PATH = app_main.DATA_DIR / "sync.json"
    app_main.init_db()
    c = AuthClient()

    r = c.get("/api/sync/export")
    assert r.status_code == 200
    payload = r.json()
    assert "users" in payload and "counts" in payload

    r = c.post("/api/sync/import", json=payload)
    assert r.status_code == 200
    assert all(v == 0 for v in r.json()["merged"].values()), "全量重导应全部跳过"

    r = c.post("/api/sync/push")
    assert r.status_code == 400, "未配置同步时 push 应 400"
    r = c.post("/api/sync/pull")
    assert r.status_code == 400
