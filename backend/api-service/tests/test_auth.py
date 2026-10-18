"""认证中间件与登录端点测试"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from middleware.auth import AuthMiddleware, create_access_token
from services.config import AppSettings
from services.rate_limit import reset_rate_limits

TEST_SECRET = "test-secret-key-with-at-least-32-characters-xx"


def _build_app(enabled: bool) -> FastAPI:
    app = FastAPI()
    middleware = AuthMiddleware(
        exempt_paths={"/api/auth/login"},
        enabled=enabled,
    )

    @app.middleware("http")
    async def _auth(request, call_next):
        return await middleware(request, call_next)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/auth/login")
    async def login():
        return {"ok": True}

    @app.get("/api/chat/sessions")
    async def sessions():
        return {"data": []}

    return app


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    reset_rate_limits()
    yield
    reset_rate_limits()


class TestAuthMiddleware:
    def test_disabled_allows_all_without_token(self):
        client = TestClient(_build_app(enabled=False))
        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200

    def test_enabled_missing_token_returns_401(self):
        client = TestClient(_build_app(enabled=True))
        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 401
        assert resp.json()["error"] == "Unauthorized"

    def test_enabled_garbage_token_returns_401(self):
        client = TestClient(_build_app(enabled=True))
        resp = client.get(
            "/api/chat/sessions",
            headers={"Authorization": "Bearer not-a-jwt"},
        )
        assert resp.status_code == 401

    def test_enabled_valid_token_passes(self):
        token = create_access_token({"sub": "admin", "role": "admin"})
        client = TestClient(_build_app(enabled=True))
        resp = client.get(
            "/api/chat/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_options_preflight_not_blocked_by_auth(self):
        client = TestClient(_build_app(enabled=True))
        resp = client.options("/api/chat/sessions")
        # 认证层放行后由路由/CORS 层决定（本小应用无 CORS 中间件，405 即证明未被认证拦截）
        assert resp.status_code != 401

    def test_exempt_paths_open_but_only_exact_match(self):
        client = TestClient(_build_app(enabled=True))
        # 白名单中的公开端点
        assert client.get("/health").status_code == 200
        assert client.post("/api/auth/login").status_code == 200
        # startswith 旧漏洞：登录路径的兄弟路径不应被豁免
        assert client.get("/api/auth/me").status_code == 401
        # 业务路由树不再被整体豁免
        assert client.get("/api/chat/sessions").status_code == 401


class TestSecurityValidation:
    def test_prod_rejects_default_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "prod")
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            AppSettings.load().validate_security()

    def test_auth_enabled_rejects_default_secret(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="AUTH_ENABLED"):
            AppSettings.load().validate_security()

    def test_auth_enabled_with_custom_secret_ok(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
        AppSettings.load().validate_security()  # 不抛异常即通过


class TestLoginEndpoint:
    def test_login_success_and_wrong_credentials(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USERNAME", "admin")
        monkeypatch.setenv("ADMIN_PASSWORD", "admin-pass")
        from main import app

        client = TestClient(app)
        ok = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-pass"},
        )
        assert ok.status_code == 200
        assert ok.json()["access_token"]

        bad = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert bad.status_code == 401

    def test_login_rate_limited_after_5_attempts(self):
        from main import app

        client = TestClient(app)
        for _ in range(5):
            resp = client.post(
                "/api/auth/login",
                json={"username": "nobody", "password": "nope"},
            )
            assert resp.status_code == 401
        # 第 6 次应被限流
        resp = client.post(
            "/api/auth/login",
            json={"username": "nobody", "password": "nope"},
        )
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
