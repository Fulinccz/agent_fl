"""
依赖注入和全局状态管理
"""

from __future__ import annotations
from typing import Optional
from agents.registry import get_agent
from agents.langgraph.conversation_graph import ConversationGraph, GraphConfig
from memory.memory_manager import MemoryManager
from rag.retriever import RAGRetriever
from logger import get_logger

logger = get_logger(__name__)

# 全局状态（延迟初始化）
_rag_retriever: Optional[RAGRetriever] = None
_memory_manager: Optional[MemoryManager] = None
_conversation_graph: Optional[ConversationGraph] = None


def get_rag_retriever() -> RAGRetriever:
    """获取 RAG 检索器（单例）"""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
        _rag_retriever.initialize_knowledge_base()
        logger.info("RAG Retriever initialized")
    return _rag_retriever


def get_memory_manager() -> MemoryManager:
    """获取记忆管理器（单例）"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(max_context_length=4000)
        logger.info("Memory Manager initialized")
    return _memory_manager


def get_conversation_graph() -> ConversationGraph:
    """获取对话图（单例）"""
    global _conversation_graph
    if _conversation_graph is None:
        agent = get_agent(provider="local", model=None)
        provider = agent.provider if hasattr(agent, 'provider') else agent

        config = GraphConfig(
            max_tokens=4000,
            temperature=0.7,
            system_prompt="你是一个专业的 AI 助手，帮助用户优化简历和解答问题。",
            enable_rag=True
        )

        _conversation_graph = ConversationGraph(
            llm_provider=provider,
            memory_manager=get_memory_manager(),
            config=config
        )
        logger.info("Conversation Graph initialized")
    return _conversation_graph
