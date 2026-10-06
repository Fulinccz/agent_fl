"""
依赖注入和全局状态管理
"""

from __future__ import annotations
from typing import Optional
from agents.registry import get_agent
from agents.skills.conversation_graph import ConversationGraph, GraphConfig
from memory.memory_manager import MemoryManager
from rag.retriever import RAGRetriever
from rag.graph.neo4j.hybrid_retriever import HybridRetriever
from rag.graph.neo4j.neo4j_client import Neo4jClient
from logger import get_logger

logger = get_logger(__name__)

# 全局状态（延迟初始化）
_rag_retriever: Optional[RAGRetriever] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_memory_manager: Optional[MemoryManager] = None
_conversation_graph: Optional[ConversationGraph] = None
_master_orchestrator = None


def get_rag_retriever() -> RAGRetriever:
    """获取 RAG 检索器（单例）"""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
        _rag_retriever.initialize_knowledge_base()
        logger.info("RAG Retriever initialized")
    return _rag_retriever


def get_hybrid_retriever() -> HybridRetriever:
    """获取混元检索器（单例）"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from services.config import AppSettings
        from services.app_config import load_app_config
        config = AppSettings.load()
        rag_cfg = load_app_config().rag

        neo4j_client = Neo4jClient(
            uri=config.neo4j_uri,
            username=config.neo4j_user,
            password=config.neo4j_password
        )

        if neo4j_client.health_check():
            logger.info("Neo4j connected, enabling graph retrieval")
            _hybrid_retriever = HybridRetriever(
                top_k=rag_cfg.hybrid_top_k,
                vector_weight=rag_cfg.vector_weight,
                graph_weight=rag_cfg.graph_weight,
                enable_vector=True,
                enable_graph=True
            )
        else:
            logger.warning("Neo4j not available, falling back to vector-only retrieval")
            _hybrid_retriever = HybridRetriever(
                top_k=rag_cfg.hybrid_top_k,
                enable_vector=True,
                enable_graph=False
            )
        logger.info("Hybrid Retriever initialized")
    return _hybrid_retriever


def get_memory_manager() -> MemoryManager:
    """获取记忆管理器（单例，SQLite 持久化——重启不丢会话）"""
    global _memory_manager
    if _memory_manager is None:
        from memory.sqlite_memory import get_memory_store
        from services.app_config import load_app_config
        max_ctx = load_app_config().memory.max_context_length
        _memory_manager = MemoryManager(
            memory_store=get_memory_store(),
            max_context_length=max_ctx
        )
        logger.info("Memory Manager initialized (sqlite persistent)")
    return _memory_manager


def get_conversation_graph() -> ConversationGraph:
    """获取对话图（单例，作为多智能体系统的降级回退；请求级 provider/model 经 chat() 透传）"""
    global _conversation_graph
    if _conversation_graph is None:
        config = GraphConfig(
            max_tokens=4000,
            temperature=0.7,
            system_prompt="你是一个专业的 AI 助手，帮助用户优化简历和解答问题。",
        )

        _conversation_graph = ConversationGraph(
            llm_provider=None,  # 运行时经 get_shared_llm 按请求解析
            memory_manager=get_memory_manager(),
            config=config
        )
        logger.info("Conversation Graph initialized")
    return _conversation_graph


def get_master_orchestrator():
    """获取主从多智能体编排器（单例）

    主流程：意图识别 → Master 规划 → 执行器执行 → ReAct 反思
    """
    global _master_orchestrator
    if _master_orchestrator is None:
        from agents.mas import MasterOrchestrator

        _master_orchestrator = MasterOrchestrator()
        logger.info("Master Orchestrator initialized")
    return _master_orchestrator
