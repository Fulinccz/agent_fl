"""全局测试夹具"""
import pytest


@pytest.fixture(autouse=True)
def _disable_langfuse_tracing(monkeypatch):
    """测试环境强制 Langfuse no-op：不向云端上报任何 span

    用空字符串占位而非 delenv——get_langfuse 内部 load_dotenv(override=False)
    会把被删除的键从 .env 重新加载；空值既视为未配置，也不会被 dotenv 覆盖。
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    from services import tracing
    tracing.reset_for_tests()
    yield
    tracing.reset_for_tests()
