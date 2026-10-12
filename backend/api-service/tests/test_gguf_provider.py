"""GGUFProvider 流式停止事件测试（不起 llama-server、不加载模型）

回归：旧实现用单个实例布尔 _stop_generation，第二个流启动时会复位标志，
导致并发流中先请求的停止失效。改为请求级 Event 集合后，stop 置位所有活动流，
流结束自动注销。
"""
import json
import threading
import time

from agents.providers.local.gguf_provider import GGUFProvider


def _make_provider():
    """绕过 __init__（它会校验模型文件并拉起 llama-server）"""
    p = object.__new__(GGUFProvider)
    p._model_name = "test.gguf"
    p._stop_events = set()
    p._stop_lock = threading.Lock()
    p._payload = lambda prompt, stream, **kw: {"stream": stream}
    return p


class _BlockedResp:
    """模拟 llama-server 流式响应：阻塞到 gate 置位后返回结束行"""

    def __init__(self, gate: threading.Event):
        self._gate = gate

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        self._gate.wait(timeout=3)
        return ["data: [DONE]"]


class _TokenResp:
    """正常返回若干 token 后结束的伪响应"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        def chunk(text):
            return "data: " + json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

        return [chunk("你好"), chunk("世界"), "data: [DONE]"]


def test_stop_sets_all_active_streams_concurrently():
    """两个并发流：stop 后两个都应结束（旧实现会因第二个流复位标志而漏停第一个）"""
    p = _make_provider()
    gate = threading.Event()
    p._post = lambda payload, timeout: _BlockedResp(gate)

    results = {}

    def run_stream(key):
        results[key] = list(p.generate_stream("测试"))

    t1 = threading.Thread(target=run_stream, args=("A",))
    t2 = threading.Thread(target=run_stream, args=("B",))
    t1.start()
    t2.start()

    # 等两个流各自注册 stop event
    time.sleep(0.3)
    assert len(p._stop_events) == 2

    # 停止：两个请求级 event 同时置位，解除响应阻塞后两个流都应中止
    p.stop_generation()
    gate.set()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert not t1.is_alive() and not t2.is_alive()
    for key in ("A", "B"):
        assert results[key][-1]["type"] == "thought"
        assert "中止" in results[key][-1]["content"]
    # 结束后事件集合清空
    assert p._stop_events == set()


def test_normal_stream_completes_and_unregisters_event():
    p = _make_provider()
    p._post = lambda payload, timeout: _TokenResp()

    chunks = list(p.generate_stream("测试"))
    tokens = [c["content"] for c in chunks if c["type"] == "token"]
    assert "".join(tokens) == "你好世界"
    # 正常结束后注销
    assert p._stop_events == set()


def test_stop_with_no_active_streams_is_noop():
    p = _make_provider()
    p.stop_generation()  # 不抛异常
