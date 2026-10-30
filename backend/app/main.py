"""AI 智能学习助手 v1 后端（朴素实现）。

v1 特意采用"能跑就行"的写法，作为期末课设重构阶段的对照基线：
- 路由 + 数据库 + 业务逻辑全部混在单个文件中（上帝模块）
- 配置硬编码为常量
- sqlite3 直连、SQL 内联在各个路由函数里
- 朴素关键词检索（无向量库、无 Embedding）

v2 重构方向（详见 docs/design.md）：分层架构 + 仓储模式 + 策略模式(AI 引擎)
+ Pydantic Settings 配置中心 + 向量检索 + SSE 流式输出。
"""

import base64
import csv
import html
import io
import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import ai_helper, apppaths, auth
from . import sync as sync_mod

# ==================== v1 硬编码配置（v2 重构目标：统一配置中心） ====================
# 便携发布后 data/ 与 .env 均位于 exe 同级目录，整个文件夹拷贝即可换机运行
DB_PATH = apppaths.app_base_dir() / "data" / "app.db"
UPLOAD_DIR = apppaths.app_base_dir() / "data" / "uploads"
DATA_DIR = apppaths.app_base_dir() / "data"
ALLOWED_EXT = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "markdown",
    ".txt": "text",
    ".markdown": "markdown",
}
MAX_UPLOAD_MB = 20
CHUNK_SIZE = 500
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(title="AI 智能学习助手 API (v1)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 数据库（v1：裸 sqlite3，每请求新连接） ====================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS materials(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
            title TEXT NOT NULL,
            course TEXT NOT NULL DEFAULT '未分类',
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_points(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
            material_id INTEGER,
            course TEXT NOT NULL DEFAULT '未分类',
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(material_id) REFERENCES materials(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
            course TEXT NOT NULL DEFAULT '未分类',
            kp_id INTEGER,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            due_date TEXT,
            status TEXT NOT NULL DEFAULT 'todo',
            created_at TEXT NOT NULL,
            FOREIGN KEY(kp_id) REFERENCES knowledge_points(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS questions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
            kp_id INTEGER,
            type TEXT NOT NULL DEFAULT 'short_answer',
            question TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'manual',
            created_at TEXT NOT NULL,
            FOREIGN KEY(kp_id) REFERENCES knowledge_points(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS practice_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
            question_id INTEGER NOT NULL,
            user_answer TEXT NOT NULL DEFAULT '',
            correct INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS wrong_book(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
            question_id INTEGER NOT NULL,
            wrong_count INTEGER NOT NULL DEFAULT 1,
            resolved INTEGER NOT NULL DEFAULT 0,
            review_marked INTEGER NOT NULL DEFAULT 0,
            last_wrong_at TEXT NOT NULL,
            UNIQUE(question_id, user_id),
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
        );
        """)
    conn.commit()
    # 顺序：先补 uuid 列（demo 插入要用）→ 保证 users.id=1 → 再做 user_id 迁移
    # （wrong_book 重建时 user_id 列有外键，必须先保证 users.id=1 存在）
    migrate_uuid(conn)
    ensure_demo_user(conn)
    migrate_legacy(conn)
    conn.close()


USER_TABLES = (
    "materials",
    "knowledge_points",
    "tasks",
    "questions",
    "practice_records",
    "wrong_book",
)


def migrate_legacy(conn: sqlite3.Connection) -> None:
    """给 v1 单用户库补 user_id 列，历史数据全部归入 id=1 的演示账号。

    幂等：每次启动都跑，已迁移的库直接跳过。
    迁移前自动备份一份 app.db.bak-pre-auth，防止 ALTER 中途断电。
    """
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    need_alter = []
    for t in USER_TABLES:
        if t not in tables:
            continue  # 新表由 CREATE TABLE 建好，无需处理
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        if "user_id" not in cols:
            need_alter.append(t)

    # wrong_book 的 UNIQUE(question_id) 必须升级为 UNIQUE(question_id, user_id)，
    # 否则两个用户答错同一道题时第二条会撞唯一约束。
    wb_sql = ""
    if "wrong_book" in tables:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='wrong_book'"
        ).fetchone()
        wb_sql = (row["sql"] if row else "") or ""
    need_rebuild = bool(wb_sql) and "UNIQUE(question_id, user_id)" not in wb_sql

    if not need_alter and not need_rebuild:
        return

    if DB_PATH.exists():
        bak = DB_PATH.with_name("app.db.bak-pre-auth")
        if not bak.exists():  # 只留最早的一份，避免覆盖掉用户仅有的备份
            shutil.copy2(DB_PATH, bak)

    for t in need_alter:
        # SQLite 限制：ALTER ADD COLUMN 带 REFERENCES 时 DEFAULT 不能为非 NULL，
        # 因此迁移列不建物理外键，user_id 作为逻辑外键指向 users(id)
        conn.execute(f"ALTER TABLE {t} ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1")
    conn.commit()

    if need_rebuild:
        conn.executescript("""
            DROP TABLE IF EXISTS wrong_book_new;  -- 上次迁移中断的残留
            CREATE TABLE wrong_book_new(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
                question_id INTEGER NOT NULL,
                wrong_count INTEGER NOT NULL DEFAULT 1,
                resolved INTEGER NOT NULL DEFAULT 0,
                review_marked INTEGER NOT NULL DEFAULT 0,
                last_wrong_at TEXT NOT NULL,
                UNIQUE(question_id, user_id),
                FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
            );
            INSERT INTO wrong_book_new(id,user_id,question_id,wrong_count,resolved,review_marked,last_wrong_at)
                SELECT id,user_id,question_id,wrong_count,resolved,review_marked,last_wrong_at FROM wrong_book;
            DROP TABLE wrong_book;
            ALTER TABLE wrong_book_new RENAME TO wrong_book;
        """)
        conn.commit()


def migrate_uuid(conn: sqlite3.Connection) -> None:
    """v1.2：给 users 与全部业务表加 uuid 身份列（跨机同步的行唯一键）。

    幂等。老行回填随机 uuid；此后所有插入路径都必须生成 uuid。
    """
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    changed = False
    for t in ("users", *USER_TABLES):
        if t not in tables:
            continue
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        if "uuid" not in cols:
            conn.execute(f"ALTER TABLE {t} ADD COLUMN uuid TEXT")
            changed = True
    conn.commit()
    # 回填空 uuid（含历史行与迁移前插入的行）
    for t in ("users", *USER_TABLES):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        if "uuid" not in cols:
            continue
        while True:
            rows = conn.execute(
                f"SELECT id FROM {t} WHERE uuid IS NULL OR uuid='' LIMIT 200"
            ).fetchall()
            if not rows:
                break
            for r in rows:
                conn.execute(f"UPDATE {t} SET uuid=? WHERE id=?", (uuid.uuid4().hex, r["id"]))
            conn.commit()
            changed = True
    # uuid 上唯一索引：同步合并按它查重
    for t in ("users", *USER_TABLES):
        idx = f"idx_{t}_uuid"
        have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        if t in tables and idx not in have:
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {idx} ON {t}(uuid)")
            changed = True
    if changed:
        conn.commit()


def ensure_demo_user(conn: sqlite3.Connection) -> None:
    """保证 id=1 的演示账号存在：历史数据全部挂在它名下。

    密码固定 demo123，登录页会提示。若用户已手动删过 demo 账号则不再复活
    （此时历史数据对所有人不可见，等于归档）。
    """
    row = conn.execute("SELECT id FROM users WHERE username='demo' OR id=1").fetchone()
    if row:
        return
    salt, digest = auth.hash_password("demo123")
    conn.execute(
        "INSERT INTO users(id,uuid,username,password_hash,salt,display_name,created_at) VALUES(1,?,'demo',?,?,'演示数据',?)",
        (uuid.uuid4().hex, digest, salt, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()


init_db()


# ==================== 认证（v1.1 新增） ====================
def require_user(authorization: str = Header(default="")) -> int:
    """所有业务接口的依赖：校验 Bearer 令牌，返回当前用户 id。"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "未登录")
    uid = auth.parse_token(authorization[7:])
    if uid is None:
        raise HTTPException(401, "登录已过期，请重新登录")
    row = db().execute("SELECT id FROM users WHERE id=?", (uid,)).fetchone()
    if not row:  # 账号被删，令牌作废
        raise HTTPException(401, "账号不存在，请重新登录")
    return uid


def _public_user(row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
    }


@app.post("/api/auth/register")
def register(body: dict):
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    display_name = (body.get("display_name") or "").strip() or username

    if not (2 <= len(username) <= 32):
        raise HTTPException(400, "用户名长度需在 2~32 个字符之间")
    if any(ch.isspace() for ch in username):
        raise HTTPException(400, "用户名不能包含空格")
    if len(password) < 6:
        raise HTTPException(400, "密码至少 6 位")

    conn = db()
    exists = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if exists:
        conn.close()
        raise HTTPException(409, "用户名已被注册")

    salt, digest = auth.hash_password(password)
    cur = conn.execute(
        "INSERT INTO users(username,uuid,password_hash,salt,display_name,created_at) VALUES(?,?,?,?,?,?)",
        (username, uuid.uuid4().hex, digest, salt, display_name, now()),
    )
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return {
        "token": auth.make_token(uid),
        "user": {"id": uid, "username": username, "display_name": display_name},
    }


@app.post("/api/auth/login")
def login(body: dict):
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    conn = db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    # 统一的失败文案，不区分「用户不存在」和「密码错误」，避免用户名枚举
    if not row or not auth.verify_password(password, row["salt"], row["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    return {"token": auth.make_token(row["id"]), "user": _public_user(row)}


@app.get("/api/auth/me")
def me(uid: int = Depends(require_user)):
    conn = db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(401, "账号不存在")
    return {"user": _public_user(row)}


@app.post("/api/auth/logout")
def logout():
    """令牌是无状态的，服务端没有可注销的东西；前端删掉本地 token 即可。

    保留这个接口是为了语义完整（前端不用特判），也便于未来升级为
    服务端黑名单时不需要改前端调用方式。
    """
    return {"ok": True}


# ==================== 文档解析 ====================
def parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_file(path: Path, ext: str) -> str:
    if ext == "pdf":
        return parse_pdf(path)
    if ext == "docx":
        return parse_docx(path)
    return path.read_text(encoding="utf-8", errors="ignore")


# ==================== 业务辅助（v1：与路由混放） ====================
def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def material_chunks(material_id: int, uid: int) -> list:
    conn = db()
    row = conn.execute(
        "SELECT content FROM materials WHERE id=? AND user_id=?", (material_id, uid)
    ).fetchone()
    conn.close()
    if not row:
        return []
    return ai_helper.chunk_text(row["content"], CHUNK_SIZE)


def course_chunks(course: str, uid: int) -> list:
    conn = db()
    rows = conn.execute(
        "SELECT content FROM materials WHERE course=? AND user_id=?", (course, uid)
    ).fetchall()
    conn.close()
    chunks = []
    for r in rows:
        chunks.extend(ai_helper.chunk_text(r["content"], CHUNK_SIZE))
    return chunks


def build_context(chunks: list, limit: int = None) -> str:
    limit = limit or MAX_CONTEXT_CHARS
    parts, total = [], 0
    for c in chunks:
        if total + len(c) > limit:
            break
        parts.append(c)
        total += len(c)
    return "\n\n".join(parts)


# ==================== 资料管理 ====================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ai_available": ai_helper.is_available(),
        "ai_base_url": ai_helper.get_base_url() if ai_helper.is_available() else None,
        "ai_model": ai_helper.get_model() if ai_helper.is_available() else None,
    }


@app.post("/api/materials/upload")
async def upload_material(
    file: UploadFile = File(...),
    course: str = Form("未分类"),
    uid: int = Depends(require_user),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext}，仅支持 PDF/DOCX/MD/TXT")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"文件超过 {MAX_UPLOAD_MB}MB 限制")

    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / save_name
    save_path.write_bytes(raw)

    try:
        content = parse_file(save_path, ALLOWED_EXT[ext])
    except Exception as e:  # v1：解析失败直接删文件报错
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"文件解析失败：{e}")
    if not content.strip():
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "解析结果为空，可能是扫描版 PDF（暂不支持 OCR）")

    conn = db()
    cur = conn.execute(
        "INSERT INTO materials(user_id,uuid,title,course,filename,filetype,content,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (
            uid,
            uuid.uuid4().hex,
            Path(file.filename).stem,
            course.strip() or "未分类",
            save_name,
            ALLOWED_EXT[ext],
            content,
            now(),
        ),
    )
    conn.commit()
    mid = cur.lastrowid
    conn.close()
    return {
        "id": mid,
        "title": Path(file.filename).stem,
        "course": course,
        "chars": len(content),
    }


@app.get("/api/materials")
def list_materials(course: str = None, uid: int = Depends(require_user)):
    conn = db()
    if course:
        rows = conn.execute(
            "SELECT id,title,course,filetype,created_at,length(content) AS chars FROM materials WHERE course=? AND user_id=? ORDER BY id DESC",
            (course, uid),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id,title,course,filetype,created_at,length(content) AS chars FROM materials WHERE user_id=? ORDER BY id DESC",
            (uid,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/materials/{mid}")
def get_material(mid: int, uid: int = Depends(require_user)):
    conn = db()
    row = conn.execute("SELECT * FROM materials WHERE id=? AND user_id=?", (mid, uid)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "资料不存在")
    return dict(row)


@app.delete("/api/materials/{mid}")
def delete_material(mid: int, uid: int = Depends(require_user)):
    conn = db()
    row = conn.execute(
        "SELECT filename FROM materials WHERE id=? AND user_id=?", (mid, uid)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "资料不存在")
    conn.execute("DELETE FROM materials WHERE id=? AND user_id=?", (mid, uid))
    conn.commit()
    conn.close()
    (UPLOAD_DIR / row["filename"]).unlink(missing_ok=True)
    return {"deleted": mid}


# ==================== 知识点 ====================
@app.post("/api/knowledge-points")
def create_kp(body: dict, uid: int = Depends(require_user)):
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "知识点标题不能为空")
    conn = db()
    cur = conn.execute(
        "INSERT INTO knowledge_points(user_id,uuid,material_id,course,title,description,created_at) VALUES(?,?,?,?,?,?,?)",
        (
            uid,
            uuid.uuid4().hex,
            body.get("material_id"),
            body.get("course", "未分类"),
            title,
            body.get("description", ""),
            now(),
        ),
    )
    conn.commit()
    kid = cur.lastrowid
    conn.close()
    return {"id": kid}


@app.get("/api/knowledge-points")
def list_kps(course: str = None, uid: int = Depends(require_user)):
    conn = db()
    sql = """SELECT k.*, m.title AS material_title,
             (SELECT COUNT(*) FROM questions q WHERE q.kp_id=k.id) AS question_count
             FROM knowledge_points k LEFT JOIN materials m ON m.id=k.material_id"""
    params = [uid]
    sql += " WHERE k.user_id=?"
    if course:
        sql += " AND k.course=?"
        params.append(course)
    sql += " ORDER BY k.id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.put("/api/knowledge-points/{kid}")
def update_kp(kid: int, body: dict, uid: int = Depends(require_user)):
    conn = db()
    fields, params = [], []
    for col in ("title", "description", "course"):
        if col in body:
            fields.append(f"{col}=?")
            params.append(body[col])
    if not fields:
        conn.close()
        raise HTTPException(400, "没有可更新字段")
    params.extend([kid, uid])
    cur = conn.execute(
        f"UPDATE knowledge_points SET {','.join(fields)} WHERE id=? AND user_id=?",
        params,
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if not changed:
        raise HTTPException(404, "知识点不存在")
    return {"updated": kid}


@app.delete("/api/knowledge-points/{kid}")
def delete_kp(kid: int, uid: int = Depends(require_user)):
    conn = db()
    cur = conn.execute("DELETE FROM knowledge_points WHERE id=? AND user_id=?", (kid, uid))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if not changed:
        raise HTTPException(404, "知识点不存在")
    return {"deleted": kid}


# ==================== 学习任务 ====================
@app.post("/api/tasks")
def create_task(body: dict, uid: int = Depends(require_user)):
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "任务标题不能为空")
    due = body.get("due_date") or None
    if due:
        try:
            datetime.fromisoformat(due)
        except ValueError:
            raise HTTPException(400, "due_date 需为 ISO 日期格式")
    conn = db()
    cur = conn.execute(
        "INSERT INTO tasks(user_id,uuid,course,kp_id,title,description,due_date,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (
            uid,
            uuid.uuid4().hex,
            body.get("course", "未分类"),
            body.get("kp_id"),
            title,
            body.get("description", ""),
            due,
            "todo",
            now(),
        ),
    )
    conn.commit()
    tid = cur.lastrowid
    conn.close()
    return {"id": tid}


@app.get("/api/tasks")
def list_tasks(uid: int = Depends(require_user)):
    conn = db()
    rows = conn.execute(
        """SELECT t.*, k.title AS kp_title,
           CASE WHEN t.due_date IS NOT NULL AND t.status!='done' AND date(t.due_date)<date('now','localtime')
                THEN 1 ELSE 0 END AS overdue
           FROM tasks t LEFT JOIN knowledge_points k ON k.id=t.kp_id
           WHERE t.user_id=? ORDER BY t.id DESC""",
        (uid,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.patch("/api/tasks/{tid}/status")
def set_task_status(tid: int, body: dict, uid: int = Depends(require_user)):
    status = body.get("status")
    if status not in ("todo", "doing", "done"):
        raise HTTPException(400, "status 只能是 todo/doing/done")
    conn = db()
    cur = conn.execute("UPDATE tasks SET status=? WHERE id=? AND user_id=?", (status, tid, uid))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if not changed:
        raise HTTPException(404, "任务不存在")
    return {"updated": tid, "status": status}


@app.delete("/api/tasks/{tid}")
def delete_task(tid: int, uid: int = Depends(require_user)):
    conn = db()
    cur = conn.execute("DELETE FROM tasks WHERE id=? AND user_id=?", (tid, uid))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if not changed:
        raise HTTPException(404, "任务不存在")
    return {"deleted": tid}


# ==================== 练习与错题 ====================
@app.post("/api/questions/generate")
def generate_questions(body: dict, uid: int = Depends(require_user)):
    """用 AI（或规则引擎降级）从知识点关联资料生成练习题。"""
    kid = body.get("kp_id")
    num = min(int(body.get("num", 5)), 10)
    conn = db()
    kp = conn.execute(
        "SELECT * FROM knowledge_points WHERE id=? AND user_id=?", (kid, uid)
    ).fetchone()
    if not kp:
        conn.close()
        raise HTTPException(404, "知识点不存在")
    content = ""
    if kp["material_id"]:
        row = conn.execute(
            "SELECT content FROM materials WHERE id=? AND user_id=?",
            (kp["material_id"], uid),
        ).fetchone()
        content = row["content"] if row else ""
    conn.close()

    if ai_helper.is_available() and content:
        try:
            raw = ai_helper.call_llm(
                "你是出题助手。根据给定资料出简答题，只输出 JSON 数组，"
                '每项 {"type":"short_answer","question":"...","answer":"..."}，'
                f"共 {num} 题，题目必须能从资料中找到答案。",
                f"知识点：{kp['title']}\n资料：\n{build_context(ai_helper.chunk_text(content, CHUNK_SIZE))}",
            )
            questions = ai_helper.parse_questions_json(raw)
        except Exception:
            questions = []
        source = "ai"
    else:
        questions = []
        source = "fallback"

    if not questions:
        questions = ai_helper.fallback_questions(kp["title"], content or kp["title"], num)
        source = "fallback"

    conn = db()
    ids = []
    for q in questions[:num]:
        cur = conn.execute(
            "INSERT INTO questions(user_id,uuid,kp_id,type,question,answer,source,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                uid,
                uuid.uuid4().hex,
                kid,
                q["type"],
                q["question"],
                q["answer"],
                source,
                now(),
            ),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return {"generated": len(ids), "source": source}


@app.get("/api/questions")
def list_questions(kp_id: int = None, uid: int = Depends(require_user)):
    conn = db()
    sql = """SELECT q.*, k.title AS kp_title FROM questions q
             LEFT JOIN knowledge_points k ON k.id=q.kp_id"""
    params = [uid]
    sql += " WHERE q.user_id=?"
    if kp_id:
        sql += " AND q.kp_id=?"
        params.append(kp_id)
    sql += " ORDER BY q.id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/practice/submit")
def submit_practice(body: dict, uid: int = Depends(require_user)):
    qid = body.get("question_id")
    correct = 1 if body.get("correct") else 0
    answer = body.get("answer", "")
    conn = db()
    q = conn.execute("SELECT id FROM questions WHERE id=? AND user_id=?", (qid, uid)).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "题目不存在")
    conn.execute(
        "INSERT INTO practice_records(user_id,uuid,question_id,user_answer,correct,created_at) VALUES(?,?,?,?,?,?)",
        (uid, uuid.uuid4().hex, qid, answer, correct, now()),
    )
    if not correct:
        conn.execute(
            """INSERT INTO wrong_book(user_id,uuid,question_id,wrong_count,last_wrong_at) VALUES(?,?,?,?,?)
               ON CONFLICT(question_id, user_id) DO UPDATE SET wrong_count=wrong_count+1,
               resolved=0, last_wrong_at=excluded.last_wrong_at""",
            (uid, uuid.uuid4().hex, qid, 1, now()),
        )
    conn.commit()
    conn.close()
    return {"recorded": True, "correct": bool(correct)}


@app.get("/api/wrong-book")
def wrong_book(only_unresolved: bool = True, uid: int = Depends(require_user)):
    conn = db()
    sql = """SELECT w.*, q.question, q.answer, q.kp_id, k.title AS kp_title,
             (SELECT COUNT(*) FROM practice_records pr WHERE pr.question_id=q.id) AS attempts
             FROM wrong_book w JOIN questions q ON q.id=w.question_id
             LEFT JOIN knowledge_points k ON k.id=q.kp_id"""
    params = [uid]
    sql += " WHERE w.user_id=?"
    if only_unresolved:
        sql += " AND w.resolved=0"
    sql += " ORDER BY w.last_wrong_at DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.patch("/api/wrong-book/{wid}")
def update_wrong(wid: int, body: dict, uid: int = Depends(require_user)):
    conn = db()
    fields, params = [], []
    if "resolved" in body:
        fields.append("resolved=?")
        params.append(1 if body["resolved"] else 0)
    if "review_marked" in body:
        fields.append("review_marked=?")
        params.append(1 if body["review_marked"] else 0)
    if not fields:
        conn.close()
        raise HTTPException(400, "没有可更新字段")
    params.extend([wid, uid])
    cur = conn.execute(f"UPDATE wrong_book SET {','.join(fields)} WHERE id=? AND user_id=?", params)
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if not changed:
        raise HTTPException(404, "错题记录不存在")
    return {"updated": wid}


def _uid_flexible(authorization: str, token_query: str = "") -> int:
    """下载/打印类路由的鉴权：优先 Bearer 头，兼容 ?token= 查询参数
    （浏览器 <a> 下载没法带 Authorization 头）。"""
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    elif token_query:
        token = token_query
    if not token:
        raise HTTPException(401, "未登录")
    uid = auth.parse_token(token)
    if uid is None:
        raise HTTPException(401, "登录已过期，请重新登录")
    return uid


def _wrong_rows(uid: int, only_unresolved: bool) -> list:
    conn = db()
    sql = """SELECT w.wrong_count, w.resolved, w.review_marked, w.last_wrong_at,
             q.question, q.answer, k.title AS kp_title
             FROM wrong_book w JOIN questions q ON q.id=w.question_id
             LEFT JOIN knowledge_points k ON k.id=q.kp_id
             WHERE w.user_id=?"""
    if only_unresolved:
        sql += " AND w.resolved=0"
    sql += " ORDER BY w.last_wrong_at DESC"
    rows = conn.execute(sql, (uid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/wrong-book/export/csv")
def export_wrong_csv(
    only_unresolved: bool = False,
    token: str = "",
    authorization: str = Header(default=""),
):
    uid = _uid_flexible(authorization, token)
    rows = _wrong_rows(uid, only_unresolved)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["知识点", "题目", "参考答案", "错误次数", "最近错误时间", "状态", "待重点复习"]
    )
    for r in rows:
        writer.writerow(
            [
                r["kp_title"] or "未关联",
                r["question"],
                r["answer"],
                r["wrong_count"],
                r["last_wrong_at"],
                "已掌握" if r["resolved"] else "未解决",
                "是" if r["review_marked"] else "",
            ]
        )
    # utf-8-sig：Excel 双击打开不乱码
    content = buf.getvalue().encode("utf-8-sig")
    filename = f"错题本-{datetime.now():%Y%m%d}.csv"
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@app.get("/api/wrong-book/print")
def print_wrong_book(
    only_unresolved: bool = False,
    token: str = "",
    authorization: str = Header(default=""),
):
    """打印友好版错题本：浏览器里 Ctrl+P 即可「另存为 PDF」。"""
    uid = _uid_flexible(authorization, token)
    rows = _wrong_rows(uid, only_unresolved)
    username = db().execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()["username"]

    cards = []
    for i, r in enumerate(rows, 1):
        marked = '<span class="mark">★ 重点</span>' if r["review_marked"] else ""
        cards.append(
            f'<div class="card"><div class="q">{i}. {html.escape(r["question"])} {marked}'
            f'<span class="meta">{html.escape(r["kp_title"] or "未关联")} · 错 {r["wrong_count"]} 次</span></div>'
            f'<div class="a">答：{html.escape(r["answer"])}</div></div>'
        )
    body = "\n".join(cards) or '<p class="empty">错题本是空的</p>'
    title = "错题（未掌握）" if only_unresolved else "错题本（全部）"
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title} · AI 学习助手</title>
<style>
body {{ font: 14px/1.7 "Microsoft YaHei", sans-serif; max-width: 800px; margin: 24px auto; color: #222; }}
h1 {{ font-size: 20px; }} .sub {{ color: #888; font-size: 12px; margin-bottom: 18px; }}
.card {{ border-bottom: 1px solid #ddd; padding: 10px 0; page-break-inside: avoid; }}
.q {{ font-weight: 600; }} .a {{ color: #2a6e2a; margin-top: 4px; }}
.meta {{ font-weight: 400; color: #888; font-size: 12px; margin-left: 8px; }}
.mark {{ color: #c60; font-weight: 400; font-size: 12px; }}
.empty {{ color: #888; }}
@media print {{ .noprint {{ display: none; }} }}
</style></head><body>
<h1>{title}</h1>
<div class="sub">{username} · 共 {len(rows)} 题 · 导出时间 {datetime.now():%Y-%m-%d %H:%M}</div>
{body}
<p class="noprint" style="margin-top:24px;color:#888">
  按 Ctrl+P 打印，目标打印机选「另存为 PDF」即可保存 PDF 文件。
  <a href="javascript:window.print()">现在打印</a>
</p>
<script>window.addEventListener('load', () => setTimeout(() => window.print(), 300));</script>
</body></html>"""
    return HTMLResponse(content=html_doc)


# ==================== 全局搜索 ====================
@app.get("/api/search")
def global_search(q: str, uid: int = Depends(require_user)):
    """跨资料/知识点/题目/任务的全局搜索，结果按类型分组。"""
    q = (q or "").strip()
    if not q:
        return {"materials": [], "knowledge_points": [], "questions": [], "tasks": []}
    like = f"%{q}%"
    conn = db()
    limit = 10

    def snip(text: str) -> str:
        text = text or ""
        pos = text.lower().find(q.lower())
        if pos < 0:
            return text[:80]
        start = max(0, pos - 40)
        end = min(len(text), pos + len(q) + 60)
        return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")

    materials = [
        {
            "id": r["id"],
            "title": r["title"],
            "course": r["course"],
            "snippet": snip(r["content"]),
        }
        for r in conn.execute(
            f"SELECT id,title,course,content FROM materials WHERE user_id=? AND (title LIKE ? OR content LIKE ?) ORDER BY id DESC LIMIT {limit}",
            (uid, like, like),
        ).fetchall()
    ]
    kps = [
        {
            "id": r["id"],
            "title": r["title"],
            "course": r["course"],
            "snippet": snip(r["description"]),
        }
        for r in conn.execute(
            f"SELECT id,title,course,description FROM knowledge_points WHERE user_id=? AND (title LIKE ? OR description LIKE ?) ORDER BY id DESC LIMIT {limit}",
            (uid, like, like),
        ).fetchall()
    ]
    questions = [
        {"id": r["id"], "kp_title": r["kp_title"], "snippet": snip(r["question"])}
        for r in conn.execute(
            f"""SELECT q.id, q.question, k.title AS kp_title FROM questions q
                LEFT JOIN knowledge_points k ON k.id=q.kp_id
                WHERE q.user_id=? AND (q.question LIKE ? OR q.answer LIKE ?) ORDER BY q.id DESC LIMIT {limit}""",
            (uid, like, like),
        ).fetchall()
    ]
    tasks = [
        {
            "id": r["id"],
            "title": r["title"],
            "status": r["status"],
            "snippet": snip(r["description"]),
        }
        for r in conn.execute(
            f"SELECT id,title,description,status FROM tasks WHERE user_id=? AND (title LIKE ? OR description LIKE ?) ORDER BY id DESC LIMIT {limit}",
            (uid, like, like),
        ).fetchall()
    ]
    conn.close()
    return {
        "materials": materials,
        "knowledge_points": kps,
        "questions": questions,
        "tasks": tasks,
    }


# ==================== 跨机同步（v1.2：GitHub 私库当枢纽） ====================
SYNC_CONFIG_PATH = DATA_DIR / "sync.json"


def _load_sync_config() -> dict:
    if not SYNC_CONFIG_PATH.is_file():
        raise HTTPException(400, "尚未配置同步：请先填写 GitHub 仓库与令牌")
    cfg = json.loads(SYNC_CONFIG_PATH.read_text(encoding="utf-8"))
    for k in ("repo", "token"):
        if not cfg.get(k):
            raise HTTPException(400, f"同步配置缺少 {k}")
    cfg.setdefault("branch", "main")
    return cfg


def _gh(cfg: dict, method: str, path: str, body: dict = None, raw: bool = False):
    """调 GitHub API，统一走 httpx，错误转 HTTPException。"""
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "ai-learning-assistant-sync",
    }
    try:
        resp = httpx.request(
            method, f"https://api.github.com{path}", headers=headers, json=body, timeout=30
        )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"GitHub 连接失败：{e}")
    if resp.status_code >= 400:
        detail = (
            resp.json().get("message", resp.text)
            if "json" in resp.headers.get("content-type", "")
            else resp.text
        )
        raise HTTPException(resp.status_code, f"GitHub 返回 {resp.status_code}：{detail}")
    if raw:
        return resp.content
    return resp.json() if resp.text else {}


@app.post("/api/sync/config")
def sync_config(body: dict, uid: int = Depends(require_user)):
    """保存同步配置：GitHub 仓库（owner/name）+ 有写权限的令牌。"""
    repo = (body.get("repo") or "").strip()
    token = (body.get("token") or "").strip()
    if not repo or "/" not in repo:
        raise HTTPException(400, "repo 需为 owner/name 格式")
    if not token:
        raise HTTPException(400, "token 不能为空")
    SYNC_CONFIG_PATH.write_text(
        json.dumps(
            {"repo": repo, "token": token, "branch": body.get("branch") or "main"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"saved": True}


@app.get("/api/sync/export")
def sync_export(uid: int = Depends(require_user)):
    """导出本库全量数据（uuid 键控）。"""
    conn = db()
    payload = sync_mod.build_export(conn)
    conn.close()
    return payload


@app.post("/api/sync/import")
def sync_import(payload: dict, uid: int = Depends(require_user)):
    """合并一份导出 payload（手动导入入口，测试与排障用）。"""
    if not isinstance(payload, dict) or "users" not in payload:
        raise HTTPException(400, "payload 格式不对：缺少 users 字段")
    conn = db()
    stats = sync_mod.apply_import(conn, payload)
    conn.close()
    return {"merged": stats}


@app.post("/api/sync/push")
def sync_push(uid: int = Depends(require_user)):
    """把本库全量快照推到 GitHub 私库 sync/<username>.json（覆盖自己那份，不会碰别人的）。"""
    cfg = _load_sync_config()
    conn = db()
    me = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()["username"]
    payload = sync_mod.build_export(conn)
    conn.close()
    import base64

    content = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode()
    path = f"sync/{me}.json"
    # 先查旧文件 sha（contents API 更新必须带 sha）
    old_sha = None
    r = httpx.get(
        f"https://api.github.com/repos/{cfg['repo']}/contents/{path}",
        headers={"Authorization": f"Bearer {cfg['token']}"},
        timeout=30,
    )
    if r.status_code == 200:
        old_sha = r.json()["sha"]
    body = {
        "message": f"sync: {me} @{datetime.now():%Y-%m-%d %H:%M}",
        "content": content,
        "branch": cfg["branch"],
    }
    if old_sha:
        body["sha"] = old_sha
    resp = _gh(cfg, "PUT", f"/repos/{cfg['repo']}/contents/{path}", body)
    return {
        "pushed": path,
        "commit": resp.get("commit", {}).get("sha", "")[:10],
        "rows": payload.get("counts", {}),
    }


@app.post("/api/sync/pull")
def sync_pull(uid: int = Depends(require_user)):
    """拉取私库 sync/ 目录下所有快照（除自己的），逐份合并入库。"""
    cfg = _load_sync_config()
    conn = db()
    me = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()["username"]
    listing = _gh(cfg, "GET", f"/repos/{cfg['repo']}/contents/sync?ref={cfg['branch']}")
    files = [f for f in listing if f["name"].endswith(".json") and f["name"] != f"{me}.json"]
    report = []
    for f in files:
        try:
            remote_payload = json.loads(base64.b64decode(f["content"]).decode("utf-8"))
        except Exception as e:
            report.append({"file": f["name"], "error": f"解析失败 {e}"})
            continue
        stats = sync_mod.apply_import(conn, remote_payload)
        report.append({"file": f["name"], "merged": stats})
    conn.close()
    if not files:
        return {"pulled": 0, "note": "远端还没有其他人的快照（或对方还没推送）"}
    return {"pulled": len(files), "detail": report}


# ==================== 统计 ====================
@app.get("/api/stats/overview")
def stats_overview(uid: int = Depends(require_user)):
    conn = db()
    per_kp = conn.execute(
        """SELECT k.id, k.title, k.course,
           COUNT(DISTINCT q.id) AS question_count,
           COUNT(pr.id) AS attempts,
           SUM(pr.correct) AS correct_count
           FROM knowledge_points k
           LEFT JOIN questions q ON q.kp_id=k.id
           LEFT JOIN practice_records pr ON pr.question_id=q.id
           WHERE k.user_id=?
           GROUP BY k.id ORDER BY k.id""",
        (uid,),
    ).fetchall()
    totals = conn.execute(
        """SELECT (SELECT COUNT(*) FROM materials WHERE user_id=?) AS materials,
           (SELECT COUNT(*) FROM knowledge_points WHERE user_id=?) AS kps,
           (SELECT COUNT(*) FROM tasks WHERE user_id=? AND status!='done') AS open_tasks,
           (SELECT COUNT(*) FROM practice_records WHERE user_id=?) AS attempts,
           (SELECT COUNT(*) FROM wrong_book WHERE user_id=? AND resolved=0) AS unresolved_wrong""",
        (uid, uid, uid, uid, uid),
    ).fetchone()
    conn.close()

    result = []
    for r in per_kp:
        attempts, correct = r["attempts"] or 0, r["correct_count"] or 0
        acc = round(correct / attempts * 100, 1) if attempts else None
        # v1 薄弱点判定：做过题且正确率 < 60% 即标记为薄弱
        result.append(
            {
                "kp_id": r["id"],
                "title": r["title"],
                "course": r["course"],
                "attempts": attempts,
                "accuracy": acc,
                "weak": attempts >= 2 and acc is not None and acc < 60,
            }
        )
    return {"totals": dict(totals), "per_kp": result}


# ==================== AI 辅助 ====================
@app.get("/api/config/ai")
def get_ai_config(uid: int = Depends(require_user)):
    """查看当前 AI 配置（key 脱敏）与可选平台预设。AI Key 是机器级配置，不按用户隔离。"""
    snapshot = ai_helper.config_snapshot()
    snapshot["presets"] = ai_helper.PROVIDER_PRESETS
    return snapshot


@app.post("/api/config/ai")
def set_ai_config(body: dict, uid: int = Depends(require_user)):
    """在软件内配置 AI：写回 exe 同级的 .env 并即时生效，无需重启。"""
    api_key = (body.get("api_key") or "").strip()
    base_url = (body.get("base_url") or ai_helper.DEFAULT_BASE_URL).strip().rstrip("/")
    model = (body.get("model") or ai_helper.DEFAULT_MODEL).strip()

    if api_key and len(api_key) < 8:
        raise HTTPException(400, "API Key 看起来不完整（长度过短），请检查是否复制齐全")
    if not base_url.startswith("http"):
        raise HTTPException(400, "接口地址必须以 http 开头")

    try:
        ai_helper.write_env_config(api_key, base_url, model)
    except OSError as e:
        raise HTTPException(500, f"配置写入失败：{e}（请确认程序所在目录可写）")

    ai_helper.apply_runtime_config(api_key, base_url, model)
    return {
        "saved": True,
        "configured": bool(api_key),
        "env_path": str(apppaths.env_file()),
        "message": "配置已保存并生效，可直接点击「检测 AI 连接」验证",
    }


@app.post("/api/ai/test")
def ai_test(uid: int = Depends(require_user)):
    """AI 连通性自检：验证 .env 中的 key / 地址 / 模型三项配置是否可用。"""
    return ai_helper.test_connection()


@app.post("/api/ai/summarize")
def ai_summarize(body: dict, uid: int = Depends(require_user)):
    mid = body.get("material_id")
    conn = db()
    row = conn.execute("SELECT * FROM materials WHERE id=? AND user_id=?", (mid, uid)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "资料不存在")
    if ai_helper.is_available():
        chunks = ai_helper.chunk_text(row["content"], CHUNK_SIZE)
        try:
            text = ai_helper.call_llm(
                "你是学习助手。请用中文对以下课程资料做结构化总结："
                "先给 3~5 条要点，再给一段 100 字以内的概览。基于资料本身，不要编造。",
                build_context(chunks),
            )
            return {"mode": "ai", "summary": text}
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"detail": f"AI 接口调用失败：{e}"})
    return {"mode": "fallback", "summary": ai_helper.fallback_summary(row["content"])}


@app.post("/api/ai/ask")
def ai_ask(body: dict, uid: int = Depends(require_user)):
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    material_id = body.get("material_id")
    chunks = material_chunks(material_id, uid) if material_id else []
    if not chunks:
        chunks = course_chunks(body.get("course", "未分类"), uid)
    if not chunks:
        # 全库兜底检索（仅限本人资料）
        conn = db()
        rows = conn.execute("SELECT content FROM materials WHERE user_id=?", (uid,)).fetchall()
        conn.close()
        chunks = ai_helper.chunk_text("\n".join(r["content"] for r in rows), CHUNK_SIZE)

    if ai_helper.is_available():
        context = build_context(ai_helper.search_chunks(question, chunks, top_k=5) or chunks)
        try:
            text = ai_helper.call_llm(
                "你是学习助手。仅依据下面提供的课程资料片段回答问题；"
                "资料中没有依据时明确说明无法从资料中得出。用中文回答，条理清晰。\n\n"
                f"资料片段：\n{context}",
                f"问题：{question}",
            )
            return {"mode": "ai", "answer": text}
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"detail": f"AI 接口调用失败：{e}"})
    return {"mode": "fallback", "answer": ai_helper.fallback_answer(question, chunks)}


@app.post("/api/ai/weak-analysis")
def ai_weak_analysis(body: dict, uid: int = Depends(require_user)):
    course = body.get("course")
    overview = stats_overview(uid)
    weak_kps = [
        k for k in overview["per_kp"] if k["weak"] and (not course or k["course"] == course)
    ]
    if not weak_kps:
        return {
            "mode": "stats",
            "weak_kps": [],
            "advice": "当前没有检测到正确率低于 60% 的薄弱知识点（做过至少 2 题）。",
        }

    weak_titles = "、".join(k["title"] for k in weak_kps)
    if ai_helper.is_available():
        chunks = course_chunks(course or "未分类", uid) or course_chunks(weak_kps[0]["course"], uid)
        try:
            text = ai_helper.call_llm(
                "你是学习教练。学生以下知识点练习正确率低于 60%，请结合课程资料给出："
                "1) 每个薄弱点可能的成因；2) 针对性的复习建议与练习计划（列出具体步骤）。",
                f"薄弱知识点：{weak_titles}\n\n课程资料片段：\n{build_context(chunks) if chunks else '（无资料）'}",
            )
            return {"mode": "ai", "weak_kps": weak_kps, "advice": text}
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"detail": f"AI 接口调用失败：{e}"})

    lines = [
        "【规则引擎分析（未配置 AI，模式：统计）】",
        f"检测到薄弱知识点：{weak_titles}",
        "建议：优先重做这些知识点的错题，再针对关联资料复习后完成一轮新练习。",
    ]
    return {"mode": "fallback", "weak_kps": weak_kps, "advice": "\n".join(lines)}


@app.get("/api/tasks/due-soon")
def due_soon(days: int = 3, uid: int = Depends(require_user)):
    """近 N 天到期的未完成任务（供看板提醒）。"""
    deadline = (datetime.now() + timedelta(days=days)).date().isoformat()
    conn = db()
    rows = conn.execute(
        """SELECT * FROM tasks WHERE user_id=? AND status!='done' AND due_date IS NOT NULL
           AND date(due_date)<=date(?) ORDER BY due_date""",
        (uid, deadline),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==================== 静态资源托管（便携发布用） ====================
# 存在前端构建产物时，由后端直接托管页面，实现"双击 exe 即用、无需 Node"。
# 开发态没有 frontend/dist 时自动跳过，仍走 Vite 开发服务器（5173）。
class _SPAStaticFiles(StaticFiles):
    """前端走 history 路由，磁盘上并没有 /practice 这样的文件。

    找不到文件就回落 index.html，交给 vue-router 自己分发；
    /api/ 下未命中的路径保持 404，避免接口错误被一份 HTML 掩盖。

    缓存策略：index.html 一律 no-cache（每次都拿最新，引用的资产文件名
    带内容哈希，构建后旧文件被删除，缓存旧 index.html 会导致白屏）；
    带哈希的 assets 允许浏览器长缓存（内容变则文件名变）。
    """

    async def get_response(self, path, scope):
        raw = (scope.get("path") or path or "").lstrip("/")
        is_index = raw == "" or raw == "index.html"
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not raw.startswith("api/"):
                response = await super().get_response("index.html", scope)
                is_index = True
            else:
                raise
        if is_index:
            response.headers["Cache-Control"] = "no-cache"
        elif raw.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


_frontend = apppaths.frontend_dist()
if _frontend is not None:
    app.mount("/", _SPAStaticFiles(directory=str(_frontend), html=True), name="frontend")
