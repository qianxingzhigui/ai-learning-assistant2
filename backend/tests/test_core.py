"""后端核心逻辑测试。

覆盖两类：
1. 纯函数单元测试（文本处理 / 规则引擎 / 统计逻辑）——不依赖 AI key 与数据库
2. API 集成测试（TestClient + 临时 SQLite）——覆盖 CRUD 与学习闭环
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DEEPSEEK_API_KEY", "")  # 测试环境强制走规则引擎

from app import ai_helper  # noqa: E402
from app import main as app_main  # noqa: E402

# ==================== 单元测试：文本处理 ====================


class TestChunkText:
    def test_长文本按块切分(self):
        text = "这是一段测试内容。" * 200
        chunks = ai_helper.chunk_text(text, chunk_size=500)
        assert len(chunks) >= 2
        assert all(len(c) <= 520 for c in chunks)
        # 不丢内容
        assert "".join(chunks).replace(" ", "")[:100] == text[:100].replace(" ", "")[:100]

    def test_短文本单块(self):
        assert ai_helper.chunk_text("短内容", chunk_size=500) == ["短内容"]

    def test_空文本(self):
        assert ai_helper.chunk_text("", chunk_size=500) == []


class TestKeywordsAndScore:
    def test_关键词提取排除停用词(self):
        kws = ai_helper.extract_keywords("机器学习是人工智能的一个分支。机器学习很重要。", top_n=5)
        assert "机器学习" in kws
        assert "一个" not in kws

    def test_相关性打分(self):
        text = "神经网络是深度学习的基础，深度学习又属于机器学习。"
        assert ai_helper.keyword_score("深度学习", text) > ai_helper.keyword_score("量子计算", text)

    def test_检索返回相关片段(self):
        chunks = ["今天天气不错", "强化学习通过奖励信号训练智能体", "晚饭吃什么"]
        hits = ai_helper.search_chunks("强化学习怎么训练", chunks, top_k=1)
        assert len(hits) == 1
        assert "强化学习" in hits[0]


# ==================== 单元测试：规则引擎 ====================


class TestFallbackEngine:
    SAMPLE = (
        "监督学习需要标注数据。无监督学习不需要标注数据。"
        "半监督学习介于两者之间。强化学习通过与环境交互学习。"
    ) * 10

    def test_抽取式总结包含关键词与句子(self):
        summary = ai_helper.fallback_summary(TestFallbackEngine.SAMPLE)
        assert "核心关键词" in summary
        assert "重点句子" in summary

    def test_答疑命中资料(self):
        chunks = ai_helper.chunk_text(TestFallbackEngine.SAMPLE, 500)
        ans = ai_helper.fallback_answer("什么是监督学习", chunks)
        assert "监督学习" in ans

    def test_答疑未命中资料(self):
        ans = ai_helper.fallback_answer("完全无关的问题量子泡沫", ["abc内容"])
        assert "没有检索到" in ans

    def test_模板出题数量正确(self):
        qs = ai_helper.fallback_questions("机器学习", TestFallbackEngine.SAMPLE, 3)
        assert len(qs) == 3
        assert all(q["question"] for q in qs)

    def test_题目JSON解析容错(self):
        raw = '```json\n[{"type":"short_answer","question":"1+1?","answer":"2"}]\n```'
        qs = ai_helper.parse_questions_json(raw)
        assert qs[0]["answer"] == "2"

    def test_题目JSON解析失败抛错(self):
        with pytest.raises(ValueError):
            ai_helper.parse_questions_json("不是JSON")


# ==================== API 集成测试（临时数据库） ====================


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """用临时目录替换数据库与上传目录，返回 TestClient。

    同时强制清空 AI key，保证测试永远走规则引擎、不会真实调用外部 API。
    """
    test_db = tmp_path / "test.db"
    test_upload = tmp_path / "uploads"
    test_upload.mkdir()
    monkeypatch.setattr(app_main, "DB_PATH", test_db)
    monkeypatch.setattr(app_main, "UPLOAD_DIR", test_upload)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")  # is_available() 视为未配置
    # 重新初始化临时库
    app_main.init_db()
    from fastapi.testclient import TestClient
    from tests.conftest import attach_auth

    with TestClient(app_main.app) as c:
        attach_auth(c)  # 注册随机用户并挂令牌，接口现在都要登录
        yield c

    # 清理：配置接口会直接改进程环境变量，避免污染其他用例
    for name in ("AI_API_KEY", "AI_BASE_URL", "AI_MODEL"):
        os.environ.pop(name, None)
    os.environ["DEEPSEEK_API_KEY"] = ""


def _upload(
    client, name="test.md", content="机器学习是人工智能的核心分支。深度学习属于机器学习。"
) -> int:
    resp = client.post(
        "/api/materials/upload",
        files={"file": (name, content.encode("utf-8"), "text/markdown")},
        data={"course": "人工智能"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


class TestMaterialAPI:
    def test_上传与查询(self, client):
        mid = _upload(client)
        resp = client.get("/api/materials")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1 and items[0]["id"] == mid

    def test_拒绝不支持的类型(self, client):
        resp = client.post(
            "/api/materials/upload",
            files={"file": ("a.exe", b"MZ...", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_删除资料(self, client):
        mid = _upload(client)
        assert client.delete(f"/api/materials/{mid}").status_code == 200
        assert client.get(f"/api/materials/{mid}").status_code == 404


class TestLearningLoop:
    def test_完整学习闭环(self, client):
        # 上传资料 → 建知识点 → 生成题目 → 答题（答错沉淀错题）→ 统计
        mid = _upload(client)
        resp = client.post(
            "/api/knowledge-points",
            json={
                "title": "机器学习",
                "course": "人工智能",
                "material_id": mid,
                "description": "概述",
            },
        )
        kid = resp.json()["id"]

        resp = client.post("/api/questions/generate", json={"kp_id": kid, "num": 3})
        assert resp.status_code == 200
        assert resp.json()["generated"] == 3

        questions = client.get("/api/questions", params={"kp_id": kid}).json()
        # 答错一题
        client.post(
            "/api/practice/submit",
            json={"question_id": questions[0]["id"], "correct": False, "answer": "错误答案"},
        )
        # 答对一题
        client.post(
            "/api/practice/submit",
            json={"question_id": questions[1]["id"], "correct": True, "answer": "正确答案"},
        )

        wrong = client.get("/api/wrong-book").json()
        assert len(wrong) == 1 and wrong[0]["question_id"] == questions[0]["id"]

        stats = client.get("/api/stats/overview").json()
        assert stats["totals"]["attempts"] == 2
        kp_stat = next(s for s in stats["per_kp"] if s["kp_id"] == kid)
        assert kp_stat["attempts"] == 2 and kp_stat["accuracy"] == 50.0

        # 标记错题已解决
        client.patch(f"/api/wrong-book/{wrong[0]['id']}", json={"resolved": True})
        assert client.get("/api/wrong-book").json() == []

    def test_无key走规则引擎总结与答疑(self, client):
        mid = _upload(client)
        r1 = client.post("/api/ai/summarize", json={"material_id": mid}).json()
        assert r1["mode"] == "fallback" and "总结" in r1["summary"]
        r2 = client.post(
            "/api/ai/ask", json={"question": "什么是机器学习", "material_id": mid}
        ).json()
        assert r2["mode"] == "fallback"


class TestTaskAPI:
    def test_任务状态流转与非法值(self, client):
        tid = client.post(
            "/api/tasks",
            json={"title": "复习第一章", "course": "人工智能", "due_date": "2099-01-01"},
        ).json()["id"]
        assert (
            client.patch(f"/api/tasks/{tid}/status", json={"status": "doing"}).json()["status"]
            == "doing"
        )
        assert (
            client.patch(f"/api/tasks/{tid}/status", json={"status": "非法值"}).status_code == 400
        )
        tasks = client.get("/api/tasks").json()
        assert tasks[0]["overdue"] == 0

    def test_近期待办(self, client):
        from datetime import datetime, timedelta

        tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
        client.post("/api/tasks", json={"title": "明天到期任务", "due_date": tomorrow})
        soon = client.get("/api/tasks/due-soon", params={"days": 3}).json()
        assert len(soon) == 1
        # 超出窗口的不算
        client.post("/api/tasks", json={"title": "远期任务", "due_date": "2099-01-01"})
        assert len(client.get("/api/tasks/due-soon", params={"days": 3}).json()) == 1


# ==================== AI 引擎配置 ====================


class TestAiConfig:
    def test_key_读取优先级(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "old-key")
        monkeypatch.delenv("AI_API_KEY", raising=False)
        assert ai_helper.get_api_key() == "old-key"  # 兼容旧变量名
        monkeypatch.setenv("AI_API_KEY", "new-key")
        assert ai_helper.get_api_key() == "new-key"  # 通用变量优先

    def test_免费平台预设齐全(self):
        presets = ai_helper.PROVIDER_PRESETS
        for name in ("zhipu", "siliconflow", "deepseek", "dashscope"):
            assert presets[name]["base_url"].startswith("http")
            assert presets[name]["model"]

    def test_未配置key时自检给出明确原因(self, monkeypatch):
        monkeypatch.delenv("AI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        result = ai_helper.test_connection()
        assert result["ok"] is False and result["reason"] == "no_key"

    def test_自检接口返回结构(self, client):
        r = client.post("/api/ai/test")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and "规则引擎" in body["message"]

    def test_健康检查暴露引擎信息(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok" and body["ai_available"] is False


class TestAiConfigAPI:
    """软件内配置 AI：写回 .env 并即时生效。"""

    def test_读取配置含预设与脱敏(self, client):
        body = client.get("/api/config/ai").json()
        assert body["configured"] is False
        assert body["presets"]["zhipu"]["model"] == "glm-4-flash"
        assert body["env_path"].endswith(".env")

    def test_key_脱敏展示(self):
        assert ai_helper.mask_key("") == ""
        assert ai_helper.mask_key("short") == "****"
        masked = ai_helper.mask_key("abcdefghijklmnop")
        assert masked.startswith("abcd") and masked.endswith("mnop") and "*" in masked

    def test_保存配置后立即生效(self, client, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        monkeypatch.setattr(ai_helper.apppaths, "env_file", lambda: env_file)

        resp = client.post(
            "/api/config/ai",
            json={
                "api_key": "test-key-1234567890",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
            },
        )
        assert resp.status_code == 200 and resp.json()["saved"] is True
        # 写入文件
        content = env_file.read_text(encoding="utf-8")
        assert "AI_API_KEY=test-key-1234567890" in content
        assert "AI_MODEL=glm-4-flash" in content
        # 立即生效（无需重启）：健康检查变为已配置
        assert client.get("/api/health").json()["ai_available"] is True
        assert client.get("/api/config/ai").json()["api_key_masked"].startswith("test")

    def test_短路或非法地址被拒绝(self, client):
        assert client.post("/api/config/ai", json={"api_key": "abc"}).status_code == 400
        assert (
            client.post(
                "/api/config/ai", json={"api_key": "abcdefghijk", "base_url": "ftp://x"}
            ).status_code
            == 400
        )

    def test_重复保存不重复追加字段(self, client, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        monkeypatch.setattr(ai_helper.apppaths, "env_file", lambda: env_file)
        for model in ("glm-4-flash", "glm-4.7-flash"):
            client.post(
                "/api/config/ai",
                json={"api_key": "abcdefghijk", "base_url": "https://x.cn/v1", "model": model},
            )
        text = env_file.read_text(encoding="utf-8")
        assert text.count("AI_MODEL=") == 1 and "glm-4.7-flash" in text
