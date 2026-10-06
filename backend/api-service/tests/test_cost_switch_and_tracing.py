"""成本开关（DeepSeek enabled=false 自动回落本地）与 Langfuse 跟踪层单元测试

外部依赖全部 mock，不发真实请求、不依赖真实 Langfuse 服务。
"""
import os
from types import SimpleNamespace

import pytest

from services import tracing as tracing_mod
from services.app_config import load_app_config, reset_app_config
from agents import llm as mas_llm
import agents.llm as llm_mod


@pytest.fixture(autouse=True)
def _reset_caches():
    """每个用例前后重置配置与 LLM 缓存，避免污染；并强制 Langfuse no-op"""
    reset_app_config()
    llm_mod._extra_llms.clear()
    tracing_mod.reset_for_tests()
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    yield
    reset_app_config()
    llm_mod._extra_llms.clear()
    tracing_mod.reset_for_tests()


# ==================== 成本开关 ====================

class TestProviderCostSwitch:
    def test_deepseek_disabled_falls_back_to_local(self, monkeypatch):
        """config.yaml 默认 enabled=false → 请求 deepseek 自动回落本地单例"""
        monkeypatch.delenv("FULIN_MODELS__DEEPSEEK__ENABLED", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        sentinel = SimpleNamespace(tag="local-singleton")
        monkeypatch.setattr(llm_mod, "_shared_llm_instance", sentinel)
        try:
            result = mas_llm.get_shared_llm(provider="deepseek", model="deepseek-flash")
            assert result is sentinel
        finally:
            llm_mod._shared_llm_instance = None

    def test_deepseek_enabled_creates_instance(self, monkeypatch):
        """enabled=true 时正常创建 DeepSeek 实例"""
        monkeypatch.setenv("FULIN_MODELS__DEEPSEEK__ENABLED", "true")
        reset_app_config()  # 重载配置使开关生效

        def fake_get_agent(provider="local", model=None):
            return SimpleNamespace(tag=f"{provider}:{model}")

        monkeypatch.setattr("agents.registry.get_agent", fake_get_agent)
        result = mas_llm.get_shared_llm(provider="deepseek", model="deepseek-flash")
        assert result.tag == "deepseek:deepseek-flash"

    def test_default_config_has_deepseek_disabled(self, tmp_path):
        """代码默认值：deepseek 不启用（成本保护）"""
        cfg = load_app_config(tmp_path / "no_such.yaml")
        assert cfg.models.deepseek.enabled is False

    def test_yaml_enables_deepseek(self, tmp_path):
        """yaml 可打开开关"""
        import textwrap
        custom = tmp_path / "custom.yaml"
        custom.write_text(textwrap.dedent("""
            models:
              deepseek:
                enabled: true
        """), encoding="utf-8")
        cfg = load_app_config(custom)
        assert cfg.models.deepseek.enabled is True


# ==================== Langfuse 跟踪层 ====================

class TestTracingNoop:
    def test_no_env_returns_none_client(self, monkeypatch):
        """未配置 key → no-op，不抛异常"""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        assert tracing_mod.is_enabled() is False
        assert tracing_mod.get_langfuse() is None

    def test_noop_span_context(self, monkeypatch):
        """未启用时 start_span 正常进出，obs 为 None，业务代码不受影响"""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        with tracing_mod.start_span("x", as_type="agent") as obs:
            assert obs is None
        # update 函数不抛异常
        tracing_mod.update_current_span(output={"a": 1})
        tracing_mod.update_current_generation(output="t")
        assert tracing_mod.get_trace_url() is None

    def test_partial_key_disables(self, monkeypatch):
        """只配 public_key 不配 secret_key → 禁用"""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        assert tracing_mod.is_enabled() is False


class TestTracingWithMockClient:
    def _fake_client(self):
        """模拟 Langfuse v4 客户端，记录 observation 嵌套"""
        events = []

        class _Obs:
            def __init__(self, name, as_type):
                self.name, self.as_type = name, as_type

        class _CM:
            def __init__(self, name, as_type):
                self._obs = _Obs(name, as_type)

            def __enter__(self):
                events.append(("enter", self._obs.name, self._obs.as_type))
                return self._obs

            def __exit__(self, *exc):
                events.append(("exit", self._obs.name))
                return False

        client = SimpleNamespace(
            start_as_current_observation=lambda name, as_type="span", **kw: _CM(name, as_type),
            update_current_span=lambda **kw: events.append(("span_update", kw.get("output"))),
            update_current_generation=lambda **kw: events.append(("gen_update", kw.get("output"), kw.get("usage_details"))),
            get_trace_url=lambda: "http://lf/trace/1",
            shutdown=lambda: None,
        )
        return client, events

    def test_nested_spans_and_generation(self, monkeypatch):
        """agent → span / generation 嵌套调用全部转发到客户端"""
        client, events = self._fake_client()
        monkeypatch.setattr(tracing_mod, "_client", client)
        monkeypatch.setattr(tracing_mod, "_inited", True)

        with tracing_mod.start_span("mas-pipeline", as_type="agent", input="hi") as obs:
            assert obs is not None
            with tracing_mod.start_span("step:chat", input={"a": 1}):
                tracing_mod.update_current_span(output={"success": True})
            with tracing_mod.start_span("llm", as_type="generation"):
                tracing_mod.update_current_generation(
                    output="resp", usage={"input": 1, "output": 2}
                )

        enters = [(e[1], e[2]) for e in events if e[0] == "enter"]
        assert ("mas-pipeline", "agent") in enters
        assert ("step:chat", "span") in enters
        assert ("llm", "generation") in enters
        assert ("gen_update", "resp", {"input": 1, "output": 2}) in events
        assert tracing_mod.get_trace_url() == "http://lf/trace/1"

    def test_client_constructor_failure_disables(self, monkeypatch):
        """Langfuse 构造抛异常 → 降级 no-op，不影响调用方"""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")

        def _boom(**kw):
            raise RuntimeError("bad host")

        import langfuse as langfuse_mod
        monkeypatch.setattr(langfuse_mod, "Langfuse", _boom)
        assert tracing_mod.is_enabled() is False
        with tracing_mod.start_span("x"):
            pass  # 不应抛异常
