"""
依赖注入和全局状态管理
"""

from __future__ import annotations
import threading
from typing import Optional
from agents.skills.conversation_graph import ConversationGraph, GraphConfig
from memory.memory_manager import MemoryManager
from rag.retriever import RAGRetriever
from rag.graph.hybrid_retriever import HybridRetriever
from rag.graph.neo4j_client import Neo4jClient
from logger import get_logger

logger = get_logger(__name__)

# 全局状态（延迟初始化）
_rag_retriever: Optional[RAGRetriever] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_memory_manager: Optional[MemoryManager] = None
_conversation_graph: Optional[ConversationGraph] = None
_master_orchestrator = None

# 单例初始化锁：并发首请求时避免重复构造重资源对象
_singleton_lock = threading.Lock()


def get_rag_retriever() -> RAGRetriever:
    """获取 RAG 检索器（单例，双检锁防并发重复初始化）"""
    global _rag_retriever
    if _rag_retriever is None:
        with _singleton_lock:
            if _rag_retriever is None:
                retriever = RAGRetriever()
                retriever.initialize_knowledge_base()
                _rag_retriever = retriever
                logger.info("RAG Retriever initialized")
    return _rag_retriever


def get_hybrid_retriever() -> HybridRetriever:
    """获取混元检索器（单例，双检锁）"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        with _singleton_lock:
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
                    retriever = HybridRetriever(
                        top_k=rag_cfg.hybrid_top_k,
                        vector_weight=rag_cfg.vector_weight,
                        graph_weight=rag_cfg.graph_weight,
                        enable_vector=True,
                        enable_graph=True
                    )
                else:
                    logger.warning("Neo4j not available, falling back to vector-only retrieval")
                    retriever = HybridRetriever(
                        top_k=rag_cfg.hybrid_top_k,
                        enable_vector=True,
                        enable_graph=False
                    )
                _hybrid_retriever = retriever
                logger.info("Hybrid Retriever initialized")
    return _hybrid_retriever


def get_memory_manager() -> MemoryManager:
    """获取记忆管理器（单例，SQLite 持久化——重启不丢会话）"""
    global _memory_manager
    if _memory_manager is None:
        with _singleton_lock:
            if _memory_manager is None:
                from memory import get_memory_store
                from services.app_config import load_app_config
                max_ctx = load_app_config().memory.max_context_length
                manager = MemoryManager(
                    memory_store=get_memory_store(),
                    max_context_length=max_ctx
                )
                _memory_manager = manager
                logger.info("Memory Manager initialized (sqlite persistent)")
    return _memory_manager


def get_conversation_graph() -> ConversationGraph:
    """获取对话图（单例，作为多智能体系统的降级回退；请求级 provider/model 经 chat() 透传）"""
    global _conversation_graph
    if _conversation_graph is None:
        with _singleton_lock:
            if _conversation_graph is None:
                config = GraphConfig(
                    max_tokens=4000,
                    temperature=0.7,
                    system_prompt="你是一个专业的 AI 助手，帮助用户优化简历和解答问题。",
                )
                graph = ConversationGraph(
                    llm_provider=None,  # 运行时经 get_shared_llm 按请求解析
                    memory_manager=get_memory_manager(),
                    config=config
                )
                _conversation_graph = graph
                logger.info("Conversation Graph initialized")
    return _conversation_graph


def get_master_orchestrator():
    """获取主从多智能体编排器（单例，双检锁）

    主流程：意图识别 → Master 规划 → 执行器执行 → ReAct 反思
    """
    global _master_orchestrator
    if _master_orchestrator is None:
        with _singleton_lock:
            if _master_orchestrator is None:
                from agents.mas import MasterOrchestrator
                _master_orchestrator = MasterOrchestrator()
                logger.info("Master Orchestrator initialized")
    return _master_orchestrator
