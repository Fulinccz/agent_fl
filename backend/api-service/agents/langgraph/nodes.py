"""
LangGraph 节点定义

定义可复用的工作流节点，用于构建更复杂的对话流程。
"""

from typing import Dict, Any, List, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from abc import ABC, abstractmethod
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from logger import get_logger

logger = get_logger(__name__)


class BaseNode(ABC):
    """节点基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点逻辑"""
        pass
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Executing node: {self.name}")
        try:
            result = await self.execute(state)
            logger.debug(f"Node {self.name} completed")
            return result
        except Exception as e:
            logger.error(f"Node {self.name} failed: {e}")
            raise


class ChatNode(BaseNode):
    """
    聊天节点
    
    调用 LLM 生成回复
    """
    
    def __init__(
        self,
        llm_provider: Any,
        system_prompt: Optional[str] = None,
        name: str = "chat"
    ):
        super().__init__(name)
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt or "你是一个有帮助的 AI 助手。"
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成聊天回复"""
        messages = state.get("messages", [])
        
        # 构建完整提示
        prompt_parts = [f"系统：{self.system_prompt}"]
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt_parts.append(f"用户：{msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"助手：{msg.content}")
            elif isinstance(msg, SystemMessage):
                prompt_parts.append(f"系统：{msg.content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # 添加最终提示
        prompt += "\n\n助手："
        
        # 调用 LLM
        try:
            if hasattr(self.llm_provider, 'agenerate'):
                response = await self.llm_provider.agenerate(prompt)
            elif hasattr(self.llm_provider, 'generate'):
                response = self.llm_provider.generate(prompt)
            else:
                response = self.llm_provider(prompt)
            
            response_text = response if isinstance(response, str) else str(response)
            
            return {
                "messages": [AIMessage(content=response_text)],
                **{k: v for k, v in state.items() if k != "messages"}
            }
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return {
                "messages": [AIMessage(content=f"抱歉，生成回复时出错：{str(e)}")],
                **{k: v for k, v in state.items() if k != "messages"}
            }


class ToolNode(BaseNode):
    """
    工具调用节点
    
    执行工具调用并返回结果
    """
    
    def __init__(
        self,
        tools: Dict[str, Callable],
        name: str = "tools"
    ):
        super().__init__(name)
        self.tools = tools
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        # 这里简化处理，实际应该解析 LLM 的工具调用请求
        tool_calls = state.get("tool_calls", [])
        results = []
        
        for call in tool_calls:
            tool_name = call.get("name")
            tool_params = call.get("parameters", {})
            
            if tool_name in self.tools:
                try:
                    tool = self.tools[tool_name]
                    if asyncio.iscoroutinefunction(tool):
                        result = await tool(**tool_params)
                    else:
                        result = tool(**tool_params)
                    results.append({
                        "tool": tool_name,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "error": str(e),
                        "success": False
                    })
            else:
                results.append({
                    "tool": tool_name,
                    "error": f"工具 {tool_name} 不存在",
                    "success": False
                })
        
        return {
            "tool_results": results,
            **state
        }


class RAGNode(BaseNode):
    """
    RAG 检索节点
    
    执行 RAG 检索并注入上下文
    """
    
    def __init__(
        self,
        retriever: Any,
        top_k: int = 3,
        name: str = "rag"
    ):
        super().__init__(name)
        self.retriever = retriever
        self.top_k = top_k
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行 RAG 检索"""
        messages = state.get("messages", [])
        
        # 获取最后一条用户消息作为查询
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        
        if not query or not self.retriever:
            return state
        
        try:
            # 执行检索
            if hasattr(self.retriever, 'aretrieve'):
                results = await self.retriever.aretrieve(query, top_k=self.top_k)
            elif hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query, top_k=self.top_k)
            else:
                results = self.retriever(query, top_k=self.top_k)
            
            # 构建上下文
            if results:
                context_parts = []
                for i, result in enumerate(results, 1):
                    content = result.get("content", "") if isinstance(result, dict) else str(result)
                    context_parts.append(f"[参考{i}] {content}")
                
                rag_context = "\n\n".join(context_parts)
                
                # 更新状态中的上下文
                context = state.get("context", {})
                context["rag_context"] = rag_context
                
                return {
                    **state,
                    "context": context
                }
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
        
        return state


class MemoryNode(BaseNode):
    """
    记忆节点
    
    加载和保存对话历史
    """
    
    def __init__(
        self,
        memory_manager: Any,
        mode: str = "load",  # 'load' 或 'save'
        name: str = "memory"
    ):
        super().__init__(name)
        self.memory_manager = memory_manager
        self.mode = mode
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行记忆操作"""
        session_id = state.get("session_id")
        
        if not session_id:
            return state
        
        if self.mode == "load":
            # 加载历史
            history = await self.memory_manager.get_conversation_context(session_id)
            
            # 转换为消息格式
            messages = []
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
            
            # 合并当前消息
            current_messages = state.get("messages", [])
            all_messages = messages + list(current_messages)
            
            return {
                **state,
                "messages": all_messages
            }
            
        elif self.mode == "save":
            # 保存对话
            messages = state.get("messages", [])
            
            # 找到用户和助手消息对
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    # 检查下一条是否是助手回复
                    if i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage):
                        # 已经保存过了，跳过
                        pass
                    else:
                        # 保存用户消息
                        await self.memory_manager.add_user_message(
                            session_id, msg.content
                        )
                elif isinstance(msg, AIMessage):
                    # 保存助手消息
                    await self.memory_manager.add_assistant_message(
                        session_id, msg.content
                    )
            
            return state
        
        return state


import asyncio
