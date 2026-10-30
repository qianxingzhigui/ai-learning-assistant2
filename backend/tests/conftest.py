"""测试共享设施：自动注册并登录的 TestClient。

引入账号体系后，除 /api/health 与 /api/auth/* 之外的所有接口都要求
Bearer 令牌。让每个测试拿到一个互不相干的随机用户，既避免用例之间
数据串味，也天然覆盖了「用户隔离」这条正确性约束。

用法（在测试文件里）::

    from fastapi.testclient import TestClient   # 保持原 import 不动也可
    from tests.conftest import AuthClient

    def test_xxx():
        c = AuthClient()          # 已登录的客户端
        c.get("/api/materials")   # 自动带 Authorization 头

或直接使用 fixture::

    def test_xxx(client):         # pytest fixture，同样已登录
        ...
"""

import uuid

import pytest
from app.main import app
from fastapi.testclient import TestClient


class AuthClient(TestClient):
    """注册一个随机用户并把令牌挂到默认请求头上。"""

    def __init__(self, **kwargs):
        super().__init__(app, **kwargs)
        self.username = f"t{uuid.uuid4().hex[:12]}"
        resp = self.post(
            "/api/auth/register",
            json={"username": self.username, "password": "test123456"},
        )
        assert resp.status_code == 200, f"测试用户注册失败: {resp.text}"
        data = resp.json()
        self.user_id = data["user"]["id"]
        self.user = data["user"]
        # TestClient 的默认请求头对所有后续请求生效
        self.headers.update({"Authorization": f"Bearer {data['token']}"})


@pytest.fixture
def client():
    c = AuthClient()
    yield c


def attach_auth(c) -> None:
    """给一个已存在的 TestClient 注册随机用户并挂上默认 Authorization 头。

    供 test_core / test_contract 里自建 client（临时库）的 fixture 复用，
    避免两套注册逻辑漂移。
    """
    username = f"t{uuid.uuid4().hex[:12]}"
    resp = c.post(
        "/api/auth/register",
        json={"username": username, "password": "test123456"},
    )
    assert resp.status_code == 200, f"测试用户注册失败: {resp.text}"
    data = resp.json()
    c.user_id = data["user"]["id"]
    c.user = data["user"]
    c.headers.update({"Authorization": f"Bearer {data['token']}"})
