"""
LangGraph 对话工作流（降级回退路径）

设计（业界惯例）：
- 状态最小化：对话历史不进图状态，由 memory_manager 动态加载（无 checkpointer，
  图本身无状态，跨轮累积问题从根上消除）
- 记忆职责归路由层：图内不写记忆，调用方统一保存，避免双写
- 请求级模型切换：chat/stream_chat 可传 provider/model，经共享 LLM 缓存解析

适配 LangGraph 1.1.9 + LangChain 1.2.15
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from memory.memory_manager import MemoryManager
from memory import get_memory_manager
from logger import get_logger
from services.tracing import start_span, update_current_span

logger = get_logger(__name__)


class ConversationState(TypedDict):
    """对话状态（单次请求内有效，无跨轮累积字段）"""
    session_id: str
    context: Dict[str, Any]
    response: str  # 生成结果（普通字段，节点间覆盖传递）


@dataclass
class GraphConfig:
    """对话图配置"""
    max_tokens: int = 4000
    temperature: float = 0.7
    system_prompt: str = "你是一个专业的 AI 助手，帮助用户优化简历和解答问题。"


class ConversationGraph:
    """
    LangGraph 对话工作流（降级回退路径）

    流程：准备上下文（历史 + RAG）→ 生成回复。持久化由调用方负责。
    """

    def __init__(
        self,
        llm_provider: Any = None,  # 可选：显式注入的模型提供者（测试用）
        memory_manager: Optional[MemoryManager] = None,
        config: Optional[GraphConfig] = None,
    ):
        self.llm_provider = llm_provider
        self.memory_manager = memory_manager or get_memory_manager()
        self.config = config or GraphConfig()

        workflow = self._build_workflow()
        self.app = workflow.compile()

        logger.info("ConversationGraph initialized (stateless)")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ConversationState)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.set_entry_point("prepare_context")
        workflow.add_edge("prepare_context", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow

    async def _prepare_context(self, state: ConversationState) -> Dict[str, Any]:
        """加载对话历史与 RAG 上下文（历史存 memory_manager，不进图状态）"""
        return {
            "session_id": state["session_id"],
            "context": state.get("context", {}),
            "response": "",
        }

    async def _generate_response(self, state: ConversationState) -> Dict[str, Any]:
        """生成回复节点"""
        session_id = state["session_id"]
        context = state.get("context", {})

        # 动态加载历史（含当前用户消息——调用方已保存）
        history = await self.memory_manager.get_conversation_context(session_id)

        messages: List[BaseMessage] = [SystemMessage(content=self.config.system_prompt)]
        if context.get("rag_context"):
            messages.append(SystemMessage(content=f"参考资料：\n{context['rag_context']}"))
        for msg in history:
            content = msg.get("content", "")
            role = msg.get("role")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        try:
            response_text = await self._call_llm(messages, state)
            logger.debug(f"Response generated: {len(response_text)} chars")
            return {"session_id": session_id, "context": context, "response": response_text}
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                "session_id": session_id, "context": context,
                "response": f"抱歉，生成回复时出错：{e}",
            }

    async def _call_llm(
        self, messages: List[BaseMessage], state: Optional[ConversationState] = None
    ) -> str:
        """调用 LLM（请求级 provider/model 优先，否则用注入的 provider）"""
        prompt = self._messages_to_string(messages)

        provider = (state or {}).get("context", {}).get("provider")
        model = (state or {}).get("context", {}).get("model")
        if provider:
            # 请求级切换：走共享 LLM 缓存（含成本开关回落）
            from agents.llm import get_shared_llm
            llm = get_shared_llm(provider=provider, model=model)
        elif self.llm_provider is not None:
            llm = self.llm_provider
        else:
            from agents.llm import get_shared_llm
            llm = get_shared_llm(provider="local", model=None)

        if hasattr(llm, "agenerate"):
            result = await llm.agenerate(prompt)
        elif hasattr(llm, "generate"):
            result = llm.generate(prompt)
        else:
            result = llm(prompt)

        return result if isinstance(result, str) else str(result)

    def _messages_to_string(self, messages: List[BaseMessage]) -> str:
        """将消息列表转换为字符串"""
        parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"系统：{msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"用户：{msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"助手：{msg.content}")
        return "\n\n".join(parts)

    def _initial_state(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]],
    ) -> ConversationState:
        ctx = dict(context or {})
        return {"session_id": session_id, "context": ctx, "response": ""}

    async def chat(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        save_user_message: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        执行对话（不写记忆——持久化由调用方统一负责）

        Args:
            session_id: 会话 ID
            user_message: 用户消息
            context: 额外上下文（RAG 结果；provider/model 也经此透传）
            save_user_message: 兼容旧签名（调用方已保存时传 False）
            provider/model: 请求级模型切换
        """
        if save_user_message:
            await self.memory_manager.add_user_message(session_id, user_message)

        state = self._initial_state(session_id, user_message, context)
        if provider:
            state["context"]["provider"] = provider
            state["context"]["model"] = model

        result = await self.app.ainvoke(state, {"configurable": {}})
        return result.get("response") or "抱歉，我无法生成回复。"

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        save_user_message: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """流式对话（图节点整体返回，此处按字符串分块 yield 以兼容 SSE 契约）"""
        if save_user_message:
            await self.memory_manager.add_user_message(session_id, user_message)

        state = self._initial_state(session_id, user_message, context)
        if provider:
            state["context"]["provider"] = provider
            state["context"]["model"] = model

        with start_span("fallback-conversation", as_type="agent",
                         input=user_message[:300], session_id=session_id,
                         metadata={"path": "conversation_graph", "provider": provider}):
            result = await self.app.ainvoke(state, {"configurable": {}})
            response = result.get("response") or "抱歉，我无法生成回复。"
            update_current_span(output=response[:500])

        # 分块输出（保持与旧流式契约一致的 token 事件形态）
        chunk_size = 24
        for i in range(0, len(response), chunk_size):
            yield {"type": "token", "content": response[i:i + chunk_size]}
