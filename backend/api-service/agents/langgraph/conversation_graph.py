"""
LangGraph 对话工作流 (v1.1.9 适配版)

定义对话的状态管理和工作流编排。
特点：
- 简洁的状态设计
- 支持工具调用
- 集成记忆系统

适配 LangGraph 1.1.9 + LangChain 1.2.15
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Sequence
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memory.memory_manager import MemoryManager
from memory import get_memory_manager
from logger import get_logger

logger = get_logger(__name__)


class ConversationState(TypedDict):
    """
    对话状态定义
    
    LangGraph 使用 TypedDict 定义状态，
    每个节点接收并返回状态的一部分。
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    context: Dict[str, Any]  # RAG 上下文等额外信息
    metadata: Dict[str, Any]  # 元数据


@dataclass
class GraphConfig:
    """对话图配置"""
    max_tokens: int = 4000
    temperature: float = 0.7
    system_prompt: str = "你是一个专业的 AI 助手，帮助用户解决问题。"
    enable_rag: bool = True
    enable_tools: bool = False


class ConversationGraph:
    """
    LangGraph 对话工作流
    
    构建一个简单的对话流程：
    1. 准备上下文（加载历史、RAG 检索）
    2. 生成回复
    3. 保存到记忆
    """
    
    def __init__(
        self,
        llm_provider: Any,  # 模型提供者
        memory_manager: Optional[MemoryManager] = None,
        config: Optional[GraphConfig] = None,
        checkpointer: Optional[Any] = None
    ):
        """
        初始化对话图
        
        Args:
            llm_provider: LLM 提供者（如 LocalProvider）
            memory_manager: 记忆管理器
            config: 配置
            checkpointer: 状态检查点（用于持久化）
        """
        self.llm_provider = llm_provider
        self.memory_manager = memory_manager or get_memory_manager()
        self.config = config or GraphConfig()
        self.checkpointer = checkpointer or MemorySaver()
        
        # 构建工作流
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("ConversationGraph initialized")
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        
        # 定义状态图
        workflow = StateGraph(ConversationState)
        
        # 添加节点
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_memory", self._save_memory)
        
        # 定义边
        workflow.set_entry_point("prepare_context")
        workflow.add_edge("prepare_context", "generate_response")
        workflow.add_edge("generate_response", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow
    
    async def _prepare_context(self, state: ConversationState) -> Dict[str, Any]:
        """
        准备上下文节点
        
        加载对话历史、RAG 上下文等
        """
        session_id = state["session_id"]
        
        # 加载历史对话
        history = await self.memory_manager.get_conversation_context(session_id)
        
        # 转换为 LangChain 消息格式
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        
        # 添加当前用户消息（最后一条）
        current_messages = list(state["messages"])
        if current_messages:
            messages.append(current_messages[-1])
        
        logger.debug(f"Context prepared for session {session_id}: {len(messages)} messages")
        
        return {
            "messages": messages,
            "session_id": session_id,
            "context": state.get("context", {}),
            "metadata": state.get("metadata", {})
        }
    
    async def _generate_response(self, state: ConversationState) -> Dict[str, Any]:
        """
        生成回复节点
        
        调用 LLM 生成回复
        """
        messages = list(state["messages"])
        
        # 构建提示
        prompt_messages = [
            SystemMessage(content=self.config.system_prompt)
        ] + messages
        
        # 添加 RAG 上下文（如果有）
        context = state.get("context", {})
        if context.get("rag_context"):
            rag_msg = SystemMessage(content=f"参考资料：\n{context['rag_context']}")
            prompt_messages.insert(1, rag_msg)
        
        try:
            # 调用 LLM
            response_text = await self._call_llm(prompt_messages)
            
            response = AIMessage(content=response_text)
            
            logger.debug(f"Response generated: {len(response_text)} chars")
            
            return {
                "messages": [response],
                "session_id": state["session_id"],
                "context": context,
                "metadata": state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            error_msg = AIMessage(content=f"抱歉，生成回复时出错：{str(e)}")
            return {
                "messages": [error_msg],
                "session_id": state["session_id"],
                "context": context,
                "metadata": state.get("metadata", {})
            }
    
    async def _call_llm(self, messages: List[BaseMessage]) -> str:
        """
        调用 LLM
        
        适配现有的 provider 接口
        """
        # 将消息转换为字符串
        prompt = self._messages_to_string(messages)
        
        # 调用 provider 的生成方法
        if hasattr(self.llm_provider, 'agenerate'):
            result = await self.llm_provider.agenerate(prompt)
        elif hasattr(self.llm_provider, 'generate'):
            result = self.llm_provider.generate(prompt)
        else:
            result = self.llm_provider(prompt)
        
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
    
    async def _save_memory(self, state: ConversationState) -> Dict[str, Any]:
        """
        保存记忆节点
        
        将对话保存到记忆存储
        """
        session_id = state["session_id"]
        messages = state["messages"]
        
        # 找到最后一条助手消息
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                await self.memory_manager.add_assistant_message(
                    session_id=session_id,
                    content=msg.content
                )
                break
        
        logger.debug(f"Memory saved for session {session_id}")
        
        return state
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        执行对话
        
        Args:
            session_id: 会话 ID
            user_message: 用户消息
            context: 额外上下文（如 RAG 结果）
            
        Returns:
            助手回复
        """
        # 先保存用户消息
        await self.memory_manager.add_user_message(session_id, user_message)
        
        # 准备初始状态
        initial_state: ConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "session_id": session_id,
            "context": context or {},
            "metadata": {}
        }
        
        # 执行工作流
        config = {"configurable": {"thread_id": session_id}}
        result = await self.app.ainvoke(initial_state, config)
        
        # 提取回复
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "抱歉，我无法生成回复。"
    
    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        流式对话
        
        生成流式回复
        """
        # 先保存用户消息
        await self.memory_manager.add_user_message(session_id, user_message)
        
        # 准备初始状态
        initial_state: ConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "session_id": session_id,
            "context": context or {},
            "metadata": {}
        }
        
        # 执行工作流
        config = {"configurable": {"thread_id": session_id}}
        
        async for event in self.app.astream(initial_state, config):
            if "generate_response" in event:
                messages = event["generate_response"].get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        yield msg.content
