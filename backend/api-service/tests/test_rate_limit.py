"""services/rate_limit.py 内存滑动窗口限流测试"""
from unittest.mock import MagicMock

import pytest

from services import rate_limit
from services.rate_limit import check_rate_limit, reset_rate_limits


@pytest.fixture(autouse=True)
def _reset():
    reset_rate_limits()
    yield
    reset_rate_limits()


def _make_request(ip="10.0.0.1", path="/api/resume/optimize/stream", xff=None):
    req = MagicMock()
    req.client.host = ip
    req.url.path = path
    req.headers = {"x-forwarded-for": xff} if xff else {}
    return req


def test_allows_under_limit():
    req = _make_request()
    for _ in range(10):
        allowed, _, remaining = check_rate_limit(req, "/api/resume/optimize/stream")
        assert allowed
        assert remaining >= 0
    # 第 11 次超限
    allowed, retry_after, remaining = check_rate_limit(req, "/api/resume/optimize/stream")
    assert not allowed
    assert retry_after >= 1
    assert remaining == 0


def test_ips_isolated():
    req_a = _make_request(ip="10.0.0.1")
    req_b = _make_request(ip="10.0.0.2")
    for _ in range(10):
        assert check_rate_limit(req_a, "/api/resume/optimize/stream")[0]
    # A 已超限，B 仍可访问
    assert check_rate_limit(req_b, "/api/resume/optimize/stream")[0]


def test_paths_isolated():
    req = _make_request(path="/api/resume/export-docx")
    # stream 端点 10 次不影响 export 端点独立计数
    for _ in range(10):
        check_rate_limit(req, "/api/resume/optimize/stream")
    allowed, _, _ = check_rate_limit(req, "/api/resume/export-docx")
    assert allowed


def test_export_limit_is_20():
    req = _make_request(path="/api/resume/export-docx")
    for _ in range(20):
        assert check_rate_limit(req, "/api/resume/export-docx")[0]
    assert not check_rate_limit(req, "/api/resume/export-docx")[0]


def test_window_expiry_allows_again(monkeypatch):
    req = _make_request()
    t = [1000.0]
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: t[0])

    for _ in range(10):
        check_rate_limit(req, "/api/resume/optimize/stream")
    assert not check_rate_limit(req, "/api/resume/optimize/stream")[0]

    # 推进超过窗口：旧记录过期，重新放行
    t[0] += 61
    allowed, _, remaining = check_rate_limit(req, "/api/resume/optimize/stream")
    assert allowed
    assert remaining == 9  # 1 次新请求已计入


def test_unknown_client_ip_fallback(monkeypatch):
    req = MagicMock()
    req.client = None
    req.headers = {}
    req.url.path = "/api/resume/optimize/stream"
    allowed, _, _ = check_rate_limit(req, "/api/resume/optimize/stream")
    assert allowed


def test_xff_first_hop_used_as_client_ip():
    """经代理时按 X-Forwarded-For 第一个地址区分客户端"""
    req_a = _make_request(ip="172.20.0.3", xff="203.0.113.1, 172.20.0.3")
    req_b = _make_request(ip="172.20.0.3", xff="203.0.113.2, 172.20.0.3")
    for _ in range(10):
        assert check_rate_limit(req_a, "/api/resume/optimize/stream")[0]
    assert not check_rate_limit(req_a, "/api/resume/optimize/stream")[0]
    # TCP 对端相同但 XFF 不同 → 独立桶，仍放行
    assert check_rate_limit(req_b, "/api/resume/optimize/stream")[0]


def test_stale_buckets_evicted(monkeypatch):
    """过期桶应被惰性清理，避免内存随 IP 数无界增长"""
    t = [1000.0]
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: t[0])

    req = _make_request(ip="10.0.0.99")
    check_rate_limit(req, "/api/resume/optimize/stream")
    assert any("10.0.0.99" in k for k in rate_limit._buckets)

    # 越过淘汰间隔 + TTL 后再次请求，触发清理
    t[0] += 400
    other = _make_request(ip="10.0.0.100")
    check_rate_limit(other, "/api/resume/optimize/stream")
    assert not any("10.0.0.99" in k for k in rate_limit._buckets)


def test_default_limit_for_unmapped_path():
    req = _make_request(path="/api/whatever")
    allowed_count = sum(
        1 for _ in range(20) if check_rate_limit(req)[0]
    )
    # 未配置路径走默认 10 次
    assert allowed_count == 10


# ---------- check_global_rate_limit（全局兜底滑动窗口） ----------

def test_global_limit_enforced(monkeypatch):
    monkeypatch.setattr(rate_limit, "GLOBAL_LIMIT", (3, 60))
    req = _make_request()
    for _ in range(3):
        allowed, _, remaining = rate_limit.check_global_rate_limit(req)
        assert allowed
    allowed, retry_after, remaining = rate_limit.check_global_rate_limit(req)
    assert not allowed and retry_after >= 1 and remaining == 0


def test_global_buckets_isolated_from_endpoint_buckets():
    """全局限流与端点级限流分桶计数，互不叠加"""
    from services.rate_limit import check_global_rate_limit
    req = _make_request()
    for _ in range(10):
        check_rate_limit(req, "/api/resume/optimize/stream")  # 端点桶已打满
    assert check_global_rate_limit(req)[0]  # 全局桶独立，仍放行


def test_global_ips_isolated():
    from services.rate_limit import check_global_rate_limit
    req_a = _make_request(ip="10.0.0.1")
    req_b = _make_request(ip="10.0.0.2")
    # A 消耗 120 次（默认全局限额）后超限，B 不受影响
    for _ in range(rate_limit.GLOBAL_LIMIT[0]):
        assert check_global_rate_limit(req_a)[0]
    assert not check_global_rate_limit(req_a)[0]
    assert check_global_rate_limit(req_b)[0]


# ---------- main.py 全局限流中间件 ----------

@pytest.mark.asyncio
async def test_global_middleware_rate_limited_returns_429(monkeypatch):
    import main

    monkeypatch.setattr(main, "check_global_rate_limit", lambda request: (False, 30, 0))
    called = False

    async def call_next(request):
        nonlocal called
        called = True
        return MagicMock(status_code=200)

    response = await main.rate_limit_middleware(MagicMock(), call_next)
    assert response.status_code == 429
    assert called is False


@pytest.mark.asyncio
async def test_global_middleware_fail_open_on_internal_error(monkeypatch):
    """限流器自身故障不应阻断服务（fail-open）"""
    import main

    def _boom(request):
        raise RuntimeError("limiter broken")

    monkeypatch.setattr(main, "check_global_rate_limit", _boom)

    async def call_next(request):
        return MagicMock(status_code=200)

    response = await main.rate_limit_middleware(MagicMock(), call_next)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_global_middleware_passes_when_allowed(monkeypatch):
    import main

    monkeypatch.setattr(main, "check_global_rate_limit", lambda request: (True, 0, 119))
    called = False

    async def call_next(request):
        nonlocal called
        called = True
        return MagicMock(status_code=200)

    response = await main.rate_limit_middleware(MagicMock(), call_next)
    assert response.status_code == 200
    assert called is True
