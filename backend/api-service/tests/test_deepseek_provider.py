"""DeepSeekProvider 与模型切换链路单元测试

覆盖：Provider 本体（generate / generate_stream / stop）、registry 注册、
MAS 共享 LLM 多实例缓存、执行器 LLM 解析。
外部 API 均以 stub client 模拟，不发真实网络请求。
"""
from types import SimpleNamespace

import pytest

from agents.providers.deepseek import DeepSeekProvider
from agents.registry import get_agent
from agents import llm as mas_llm
from agents.mas.executors import _resolve_llm
import agents.llm as llm_mod


# ==================== stub 构造 ====================

def _make_delta(content=None, reasoning_content=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning_content)


def _make_chunk(content=None, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=_make_delta(content, reasoning))])


class _StubCompletions:
    """模拟 openai chat.completions 接口"""

    def __init__(self, response=None, stream_chunks=None, error=None):
        self._response = response
        self._stream_chunks = stream_chunks
        self._error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        if self._stream_chunks is not None:
            return iter(self._stream_chunks)
        return self._response


def _make_provider(monkeypatch, api_key="sk-test-12345678", **client_kwargs):
    """构造带 stub client 的 Provider（不发真实请求）"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    provider = DeepSeekProvider(model="deepseek-flash")
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_StubCompletions(**client_kwargs))
    )
    return provider


def _make_response(text="你好，世界"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


# ==================== Provider 属性 ====================

class TestDeepSeekProviderProps:
    def test_model_and_device(self, monkeypatch):
        p = _make_provider(monkeypatch)
        assert p.model_name == "deepseek-flash"
        assert p.device == "cloud"

    def test_api_key_masked(self, monkeypatch):
        p = _make_provider(monkeypatch, api_key="sk-abcdef1234567890")
        masked = p.api_key
        assert masked.startswith("sk-a")
        assert masked.endswith("7890")
        assert "abcdef123456" not in masked

    def test_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        p = DeepSeekProvider()
        assert p.api_key is None


# ==================== 同步生成 ====================

class TestDeepSeekGenerate:
    def test_generate_success(self, monkeypatch):
        p = _make_provider(monkeypatch, response=_make_response("  优化后的文本  "))
        assert p.generate("帮我优化") == "优化后的文本"

    def test_generate_passes_model_and_stream_false(self, monkeypatch):
        p = _make_provider(monkeypatch, response=_make_response("ok"))
        p.generate("hi", max_new_tokens=256)
        stub = p._client.chat.completions
        assert stub.calls[-1]["model"] == "deepseek-flash"
        assert stub.calls[-1]["stream"] is False
        assert stub.calls[-1]["max_tokens"] == 256

    def test_generate_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        p = DeepSeekProvider()
        with pytest.raises(RuntimeError, match="API Key"):
            p.generate("hi")

    def test_generate_api_error_wrapped(self, monkeypatch):
        p = _make_provider(monkeypatch, error=RuntimeError("401 unauthorized"))
        with pytest.raises(RuntimeError, match="DeepSeek API 调用失败"):
            p.generate("hi")

    def test_images_rejected(self, monkeypatch):
        p = _make_provider(monkeypatch, response=_make_response("x"))
        with pytest.raises(ValueError):
            p.generate("hi", images=["a.png"])
        # 流式路径对齐本地契约：yield error 事件而非抛异常
        events = list(p.generate_stream("hi", images=["a.png"]))
        assert events[0]["type"] == "error"


# ==================== 流式生成 ====================

class TestDeepSeekStream:
    def test_stream_tokens(self, monkeypatch):
        chunks = [_make_chunk(content="你"), _make_chunk(content="好")]
        p = _make_provider(monkeypatch, stream_chunks=chunks)
        events = list(p.generate_stream("hi"))
        assert events == [
            {"type": "token", "content": "你"},
            {"type": "token", "content": "好"},
        ]

    def test_stream_reasoning_as_thought(self, monkeypatch):
        chunks = [
            _make_chunk(reasoning="正在思考"),
            _make_chunk(content="答案"),
        ]
        p = _make_provider(monkeypatch, stream_chunks=chunks)
        events = list(p.generate_stream("hi"))
        assert events == [
            {"type": "thought", "content": "正在思考"},
            {"type": "token", "content": "答案"},
        ]

    def test_stream_skips_empty_choices(self, monkeypatch):
        chunks = [SimpleNamespace(choices=[]), _make_chunk(content="x")]
        p = _make_provider(monkeypatch, stream_chunks=chunks)
        events = list(p.generate_stream("hi"))
        assert events == [{"type": "token", "content": "x"}]

    def test_stream_no_key_yields_error_event(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        p = DeepSeekProvider()
        events = list(p.generate_stream("hi"))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "DEEPSEEK_API_KEY" in events[0]["content"]

    def test_stream_api_error_yields_error_event(self, monkeypatch):
        p = _make_provider(monkeypatch, error=RuntimeError("boom"))
        events = list(p.generate_stream("hi"))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "boom" in events[0]["content"]

    def test_generate_with_thoughts_alias(self, monkeypatch):
        chunks = [_make_chunk(content="hi")]
        p = _make_provider(monkeypatch, stream_chunks=chunks)
        events = list(p.generate_with_thoughts("hi"))
        assert events == [{"type": "token", "content": "hi"}]


# ==================== 停止与参数映射 ====================

class TestDeepSeekControl:
    def test_stop_generation(self, monkeypatch):
        p = _make_provider(monkeypatch, stream_chunks=[_make_chunk(content="x")])
        p.stop_generation()
        assert p._stop_event.is_set()

    def test_build_kwargs_mapping(self, monkeypatch):
        p = _make_provider(monkeypatch)
        assert p._build_kwargs(max_new_tokens=512) == {
            "max_tokens": 512,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        assert p._build_kwargs(max_tokens=64)["max_tokens"] == 64
        assert p._build_kwargs(temperature=0.5)["temperature"] == 0.5
        deep = p._build_kwargs(deepThinking=True)
        assert deep == {"extra_body": {"thinking": {"type": "enabled"}}}
        # 默认/False 显式关闭思考（deepseek-flash 默认开启，不关会吞掉 max_tokens）
        assert p._build_kwargs() == {"extra_body": {"thinking": {"type": "disabled"}}}
        assert p._build_kwargs(deepThinking=False) == {
            "extra_body": {"thinking": {"type": "disabled"}}
        }

    def test_generate_with_image_not_implemented(self, monkeypatch):
        p = _make_provider(monkeypatch)
        with pytest.raises(NotImplementedError):
            p.generate_with_image("hi", "img.png")


# ==================== 注册与切换链路 ====================

class TestModelSwitchChain:
    def test_registry_returns_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345678")
        agent = get_agent(provider="deepseek", model="deepseek-flash")
        assert type(agent).__name__ == "DeepSeekProvider"
        assert agent.model_name == "deepseek-flash"

    def test_registry_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345678")
        agent = get_agent(provider="DeepSeek")
        assert type(agent).__name__ == "DeepSeekProvider"

    def test_mas_llm_caches_per_provider(self, monkeypatch):
        """同一 (provider, model) 缓存复用；不同 model 各自实例化"""
        from services.app_config import reset_app_config
        monkeypatch.setenv("FULIN_MODELS__DEEPSEEK__ENABLED", "true")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345678")
        reset_app_config()
        llm_mod._extra_llms.clear()

        calls = []

        def fake_get_agent(provider="local", model=None):
            calls.append((provider, model))
            return SimpleNamespace(tag=f"{provider}:{model}")

        monkeypatch.setattr("agents.registry.get_agent", fake_get_agent)

        a1 = mas_llm.get_shared_llm(provider="deepseek", model="deepseek-flash")
        a2 = mas_llm.get_shared_llm(provider="deepseek", model="deepseek-flash")
        b1 = mas_llm.get_shared_llm(provider="deepseek", model="deepseek-v4-pro")

        assert a1 is a2
        assert a1 is not b1
        # deepseek-flash 只实例化一次
        assert calls.count(("deepseek", "deepseek-flash")) == 1
        llm_mod._extra_llms.clear()
        reset_app_config()

    def test_resolve_llm_prefers_injected(self, monkeypatch):
        injected = SimpleNamespace(tag="injected")
        result = _resolve_llm(
            injected, {"provider": "deepseek", "model": "deepseek-flash"}
        )
        assert result is injected

    def test_resolve_llm_uses_task_input_provider(self, monkeypatch):
        monkeypatch.setenv("FULIN_MODELS__DEEPSEEK__ENABLED", "true")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345678")
        from services.app_config import reset_app_config
        reset_app_config()
        llm_mod._extra_llms.clear()

        def fake_get_agent(provider="local", model=None):
            return SimpleNamespace(tag=f"{provider}:{model}")

        monkeypatch.setattr("agents.registry.get_agent", fake_get_agent)
        result = _resolve_llm(
            None, {"provider": "deepseek", "model": "deepseek-flash"}
        )
        assert result.tag == "deepseek:deepseek-flash"
        llm_mod._extra_llms.clear()
        reset_app_config()

    def test_resolve_llm_default_local_passthrough(self, monkeypatch):
        """未指定 provider 时走默认本地路径（复用全局单例，不查 extra 缓存）"""
        sentinel = SimpleNamespace(tag="local-singleton")
        orig = llm_mod._shared_llm_instance
        llm_mod._shared_llm_instance = sentinel
        try:
            result = _resolve_llm(None, {"resume": "x"})
            assert result is sentinel
        finally:
            llm_mod._shared_llm_instance = orig
