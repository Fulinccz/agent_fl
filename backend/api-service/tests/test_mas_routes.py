"""多智能体路由层测试

覆盖 api/routes/chat.py 与 api/routes/resume.py 的 MAS 主流程接入：
- MAS 成功：事件/回复透传，用户消息与助手消息各保存一次
- MAS 失败：自动降级 ConversationGraph / 旧版 resume workflow（防重复保存）
- RAG 上下文构建的关闭/异常/成功路径
- deps.get_master_orchestrator 全局单例

全部依赖均被 mock，不加载真实模型、不起真实 HTTP 服务。
"""
import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from api.routes import chat as chat_route
from api.routes import resume as resume_route
from api.routes.common import ChatRequest, ResumeOptimizeRequest
from api.routes import deps
from agents.mas.state import INTENT_RESUME_OPTIMIZE


# ============ 测试夹具 ============

@pytest.fixture
def fake_memory():
    mem = MagicMock()
    mem.add_user_message = AsyncMock()
    mem.add_assistant_message = AsyncMock()
    mem.memory_store = MagicMock()
    mem.memory_store.get_history = AsyncMock(return_value=[])
    return mem


@pytest.fixture
def fake_request():
    req = MagicMock()
    req.is_disconnected = AsyncMock(return_value=False)
    return req


def _orch_ok(answer="多智能体回复"):
    """正常 MAS 编排器"""
    orch = MagicMock()
    orch.run = MagicMock(return_value={
        "success": True, "answer": answer,
        "intent": {"name": "chat-general"},
    })

    def _stream(user_input, context=None, preset_intent=None):
        yield {"type": "intent", "data": {"name": "chat-general"}}
        yield {"type": "token", "content": "多智能体"}
        yield {"type": "token", "content": "回复"}
        yield {"type": "complete", "data": {"answer": answer}}

    orch.run_stream = MagicMock(side_effect=_stream)
    return orch


async def _drain_stream(response):
    """收集 StreamingResponse 的 SSE 行并解析为字典列表"""
    events = []
    async for line in response.body_iterator:
        events.append(json.loads(line))
    return events


# ============ /chat 非流式 ============

class TestChatRoute:
    async def test_chat_mas_success(self, monkeypatch, fake_memory):
        orch = _orch_ok()
        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat(ChatRequest(message="你好"))

        assert response.response == "多智能体回复"
        assert response.sessionId  # 自动生成会话
        # 用户消息与助手消息各保存一次
        fake_memory.add_user_message.assert_awaited_once()
        fake_memory.add_assistant_message.assert_awaited_once()
        # MAS 被调用，未触发降级
        orch.run.assert_called_once()

    async def test_chat_mas_exception_falls_back_to_graph(self, monkeypatch, fake_memory):
        orch = MagicMock()
        orch.run = MagicMock(side_effect=RuntimeError("编排器挂了"))

        graph = MagicMock()
        graph.chat = AsyncMock(return_value="降级回复")

        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "get_conversation_graph", lambda: graph)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat(ChatRequest(message="你好", sessionId="s1"))

        assert response.response == "降级回复"
        # 降级调用携带 save_user_message=False（路由层已保存过用户消息）
        graph.chat.assert_awaited_once()
        assert graph.chat.await_args.kwargs["save_user_message"] is False
        fake_memory.add_user_message.assert_awaited_once()

    async def test_chat_mas_empty_answer_falls_back(self, monkeypatch, fake_memory):
        """MAS 返回 success=True 但 answer 为空时也应降级"""
        orch = MagicMock()
        orch.run = MagicMock(return_value={"success": True, "answer": ""})
        graph = MagicMock()
        graph.chat = AsyncMock(return_value="兜底回复")

        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "get_conversation_graph", lambda: graph)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat(ChatRequest(message="嗨"))
        assert response.response == "兜底回复"


# ============ /chat/stream 流式 ============

class TestChatStreamRoute:
    async def test_stream_passes_through_mas_events(self, monkeypatch, fake_memory,
                                                     fake_request):
        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: _orch_ok())
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat_stream(
            fake_request, ChatRequest(message="你好", sessionId="s1")
        )
        events = await _drain_stream(response)

        types = [e["type"] for e in events]
        assert types[0] == "session"
        assert types[-1] == "complete"
        assert "intent" in types
        # token 拼接
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "多智能体回复"
        # complete 携带 sessionId / messageCount
        assert events[-1]["sessionId"] == "s1"
        assert "messageCount" in events[-1]
        # 助手回复按累积内容保存一次
        saved = fake_memory.add_assistant_message.await_args
        assert saved.args[1] == "多智能体回复"

    async def test_stream_mas_exception_falls_back(self, monkeypatch, fake_memory,
                                                   fake_request):
        def _boom(*a, **k):
            raise RuntimeError("流炸了")
            yield  # pragma: no cover - 使其成为生成器函数

        orch = MagicMock()
        orch.run_stream = MagicMock(side_effect=_boom)

        async def _graph_stream(session_id, user_message, context, save_user_message,
                                provider=None, model=None):
            assert save_user_message is False
            for tok in ["降级", "token"]:
                yield tok

        graph = MagicMock()
        graph.stream_chat = _graph_stream

        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_conversation_graph", lambda: graph)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat_stream(
            fake_request, ChatRequest(message="你好", sessionId="s2")
        )
        events = await _drain_stream(response)

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "降级token"
        types = [e["type"] for e in events]
        assert types[-1] == "complete"
        # 用户消息仅保存一次（MAS 路径保存，降级图不重复保存）
        fake_memory.add_user_message.assert_awaited_once()


# ============ /resume/optimize/stream ============

class TestResumeRoute:
    async def test_resume_uses_preset_intent_and_passes_events(
        self, monkeypatch, fake_request
    ):
        captured = {}
        orch = MagicMock()

        def _stream(user_input, context, preset_intent):
            captured["preset"] = preset_intent
            captured["context"] = context
            yield {"type": "intent", "data": {"name": preset_intent.name}}
            yield {"type": "score", "data": {"overall_score": {"score": 80}}}
            yield {"type": "polished", "data": {"optimized_resume": "专业简历",
                                                "partial": False}}
            yield {"type": "complete", "data": {"steps_success": 3}}

        orch.run_stream = MagicMock(side_effect=_stream)
        monkeypatch.setattr(deps, "get_master_orchestrator", lambda: orch)

        response = await resume_route.resume_optimize_stream(
            fake_request,
            ResumeOptimizeRequest(resume="简历原文", jd="岗位JD", position_type="后端"),
        )
        events = await _drain_stream(response)

        # 预置意图：跳过意图识别
        assert captured["preset"].name == INTENT_RESUME_OPTIMIZE
        assert captured["preset"].source == "preset"
        assert captured["context"]["resume"] == "简历原文"
        assert captured["context"]["jd"] == "岗位JD"
        # 旧契约事件透传
        types = [e["type"] for e in events]
        assert types == ["intent", "score", "polished", "complete"]

    async def test_resume_mas_failure_falls_back_to_legacy(
        self, monkeypatch, fake_request
    ):
        # 编排器获取即失败
        monkeypatch.setattr(
            deps, "get_master_orchestrator",
            lambda: (_ for _ in ()).throw(RuntimeError("MAS 不可用")),
        )

        workflow = MagicMock()
        workflow.optimize_stream.return_value = iter([
            {"type": "score", "data": {"overall_score": {"score": 70}}},
            {"type": "complete", "data": {}},
        ])
        monkeypatch.setattr(
            "agents.skills.resume_agents.get_resume_workflow", lambda: workflow
        )

        response = await resume_route.resume_optimize_stream(
            fake_request, ResumeOptimizeRequest(resume="简历")
        )
        events = await _drain_stream(response)

        types = [e["type"] for e in events]
        assert types == ["score", "complete"]
        workflow.optimize_stream.assert_called_once()

    async def test_resume_partial_results_no_fallback(self, monkeypatch, fake_request):
        """MAS 迭代中途异常但已产出结果事件 → 不触发降级（避免重复执行）"""
        orch = MagicMock()

        def _stream(user_input, context, preset_intent):
            yield {"type": "score", "data": {"overall_score": {"score": 80}}}
            raise RuntimeError("中途断了")

        orch.run_stream = MagicMock(side_effect=_stream)
        monkeypatch.setattr(deps, "get_master_orchestrator", lambda: orch)

        # 若错误地触发降级会调用此函数，令其显形
        monkeypatch.setattr(
            "agents.skills.resume_agents.get_resume_workflow",
            lambda: (_ for _ in ()).throw(AssertionError("不应触发降级")),
        )

        response = await resume_route.resume_optimize_stream(
            fake_request, ResumeOptimizeRequest(resume="简历")
        )
        events = await _drain_stream(response)

        types = [e["type"] for e in events]
        assert types == ["score"]

    async def test_resume_legacy_inner_error_emits_error(self, monkeypatch, fake_request):
        """MAS 失败触发降级，但旧工作流也异常 → 下发 error 事件"""
        monkeypatch.setattr(
            deps, "get_master_orchestrator",
            lambda: (_ for _ in ()).throw(RuntimeError("MAS 不可用")),
        )
        workflow = MagicMock()
        workflow.optimize_stream.side_effect = RuntimeError("旧流程也挂了")
        monkeypatch.setattr(
            "agents.skills.resume_agents.get_resume_workflow", lambda: workflow
        )

        response = await resume_route.resume_optimize_stream(
            fake_request, ResumeOptimizeRequest(resume="简历")
        )
        events = await _drain_stream(response)

        assert events[-1]["type"] == "error"
        assert "旧流程也挂了" in events[-1]["message"]


# ============ RAG 上下文构建 ============

class TestBuildRagContext:
    async def test_disabled_returns_empty(self):
        ctx = await chat_route._build_rag_context("问题", False, True, "hybrid")
        assert ctx == {}

    async def test_retriever_exception_returns_empty(self, monkeypatch):
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("检索器挂了")
        monkeypatch.setattr(chat_route, "get_rag_retriever", lambda: retriever)

        ctx = await chat_route._build_rag_context("问题", True, False, "hybrid")
        assert ctx == {}

    async def test_empty_results_returns_empty(self, monkeypatch):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        monkeypatch.setattr(chat_route, "get_rag_retriever", lambda: retriever)

        ctx = await chat_route._build_rag_context("问题", True, False, "hybrid")
        assert ctx == {}

    async def test_vector_and_graph_paths(self, monkeypatch):
        # 向量检索路径（enable_graph=False）
        vector_retriever = MagicMock()
        vector_retriever.retrieve.return_value = [
            {"content": "向量片段", "sources": ["vector"]},
        ]
        monkeypatch.setattr(chat_route, "get_rag_retriever", lambda: vector_retriever)
        ctx = await chat_route._build_rag_context("q", True, False, "hybrid")
        assert "[参考1|vector] 向量片段" in ctx["rag_context"]

        # 图谱混合检索路径（enable_graph=True，携带 mode）
        hybrid_retriever = MagicMock()
        hybrid_retriever.retrieve.return_value = [
            {"content": "图谱片段", "sources": ["graph", "vector"]},
        ]
        monkeypatch.setattr(chat_route, "get_hybrid_retriever", lambda: hybrid_retriever)
        ctx = await chat_route._build_rag_context("q", True, True, "merge")
        assert "[参考1|graph+vector] 图谱片段" in ctx["rag_context"]
        hybrid_retriever.retrieve.assert_called_once_with("q", top_k=3, mode="merge")


# ============ deps 单例 ============

class TestMasterOrchestratorSingleton:
    def test_get_master_orchestrator_is_singleton(self, monkeypatch):
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)
        monkeypatch.setattr(deps, "_master_orchestrator", None)
        with patch("agents.mas.MasterOrchestrator", ctor):
            first = deps.get_master_orchestrator()
            second = deps.get_master_orchestrator()
        assert first is sentinel and second is sentinel
        ctor.assert_called_once()
        # 清理全局，避免污染其他测试
        monkeypatch.setattr(deps, "_master_orchestrator", None)


# ============ 模型切换（provider/model 透传）============

class TestModelSwitchPassthrough:
    async def test_chat_stream_passes_provider_to_mas(self, monkeypatch, fake_memory,
                                                      fake_request):
        """chat/stream 携带 provider 时应透传到 MAS context"""
        captured = {}
        orch = MagicMock()

        def _stream(user_input, context=None, preset_intent=None):
            captured["context"] = context
            yield {"type": "token", "content": "ok"}
            yield {"type": "complete", "data": {"answer": "ok"}}

        orch.run_stream = MagicMock(side_effect=_stream)
        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat_stream(
            fake_request,
            ChatRequest(message="你好", provider="deepseek", model="deepseek-flash"),
        )
        events = await _drain_stream(response)
        assert any(e["type"] == "token" for e in events)
        assert captured["context"]["provider"] == "deepseek"
        assert captured["context"]["model"] == "deepseek-flash"

    async def test_chat_stream_without_provider_has_no_key(self, monkeypatch,
                                                           fake_memory, fake_request):
        """未指定 provider 时 context 不含 provider/model 键"""
        captured = {}
        orch = MagicMock()

        def _stream(user_input, context=None, preset_intent=None):
            captured["context"] = context
            yield {"type": "token", "content": "ok"}

        orch.run_stream = MagicMock(side_effect=_stream)
        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat_stream(
            fake_request, ChatRequest(message="你好")
        )
        await _drain_stream(response)
        assert "provider" not in captured["context"]
        assert "model" not in captured["context"]

    async def test_resume_passes_provider_to_mas(self, monkeypatch, fake_request):
        """resume/optimize/stream 携带 provider 时应透传到 MAS context"""
        captured = {}
        orch = MagicMock()

        def _stream(user_input, context, preset_intent):
            captured["context"] = context
            yield {"type": "score", "data": {"overall_score": {"score": 80}}}

        orch.run_stream = MagicMock(side_effect=_stream)
        monkeypatch.setattr(deps, "get_master_orchestrator", lambda: orch)

        response = await resume_route.resume_optimize_stream(
            fake_request,
            ResumeOptimizeRequest(resume="简历", provider="deepseek"),
        )
        await _drain_stream(response)
        assert captured["context"]["provider"] == "deepseek"
        assert "model" not in captured["context"]

    async def test_chat_stream_empty_mas_answer_falls_back(self, monkeypatch,
                                                           fake_memory, fake_request):
        """MAS 正常结束但零 token → 仍降级 ConversationGraph 兜底"""
        orch = MagicMock()

        def _stream(user_input, context=None, preset_intent=None):
            yield {"type": "intent", "data": {"name": "chat-general"}}

        orch.run_stream = MagicMock(side_effect=_stream)
        graph = MagicMock()
        graph.stream_chat = _fake_graph_stream()

        monkeypatch.setattr(chat_route, "get_master_orchestrator", lambda: orch)
        monkeypatch.setattr(chat_route, "get_conversation_graph", lambda: graph)
        monkeypatch.setattr(chat_route, "get_memory_manager", lambda: fake_memory)
        monkeypatch.setattr(chat_route, "_build_rag_context",
                            AsyncMock(return_value={}))

        response = await chat_route.chat_stream(
            fake_request, ChatRequest(message="你好", sessionId="s-empty")
        )
        events = await _drain_stream(response)
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "兜底"


def _fake_graph_stream():
    async def _stream(session_id, user_message, context, save_user_message,
                      provider=None, model=None):
        assert save_user_message is False
        yield "兜底"
    return _stream
