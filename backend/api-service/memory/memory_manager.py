"""
记忆管理器 - 高级对话记忆管理

提供对话记忆的统一管理，包括：
- 会话生命周期管理
- 记忆摘要（当对话过长时生成摘要）
- 多会话切换
"""

import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from .simple_memory import SimpleMemoryStore, ConversationMemory, get_simple_memory_store
from logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    对话记忆管理器
    
    管理多个对话会话，支持：
    - 创建/切换会话
    - 自动会话清理
    - 记忆摘要生成
    """
    
    def __init__(
        self,
        memory_store: Optional[SimpleMemoryStore] = None,
        max_context_length: int = 4000,  # 最大上下文长度（字符数）
        enable_summarization: bool = True  # 是否启用自动摘要
    ):
        """
        初始化记忆管理器
        
        Args:
            memory_store: 记忆存储实例
            max_context_length: 最大上下文长度，超过则触发摘要
            enable_summarization: 是否启用自动摘要
        """
        self.memory_store = memory_store or get_simple_memory_store()
        self.max_context_length = max_context_length
        self.enable_summarization = enable_summarization
        
        # 活跃会话缓存
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"MemoryManager initialized: max_context={max_context_length}")
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建新会话
        
        Args:
            session_id: 可选的会话 ID，不传则自动生成
            metadata: 会话元数据
            
        Returns:
            会话 ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self._active_sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'summary': None
        }
        
        logger.info(f"Session created: {session_id}")
        return session_id
    
    async def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加用户消息"""
        await self.memory_store.add_message(
            session_id=session_id,
            role='user',
            content=content,
            metadata=metadata
        )
    
    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加助手消息"""
        await self.memory_store.add_message(
            session_id=session_id,
            role='assistant',
            content=content,
            metadata=metadata
        )
    
    async def get_conversation_context(
        self,
        session_id: str,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        获取对话上下文，用于构建 LLM 提示
        
        Args:
            session_id: 会话 ID
            include_summary: 是否包含会话摘要
            
        Returns:
            格式化的对话历史
        """
        messages = await self.memory_store.get_history_as_messages(session_id)
        
        # 如果启用了摘要且上下文很长，添加摘要
        if self.enable_summarization and include_summary:
            total_length = sum(len(m['content']) for m in messages)
            
            if total_length > self.max_context_length:
                # 生成或获取摘要
                summary = await self._get_or_create_summary(session_id, messages)
                
                # 返回摘要 + 最近的几条消息
                recent_messages = messages[-4:]  # 最近 2 轮对话
                context = [
                    {'role': 'system', 'content': f'这是之前对话的摘要：{summary}'}
                ] + recent_messages
                
                return context
        
        return messages
    
    async def _get_or_create_summary(
        self,
        session_id: str,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        获取或创建会话摘要
        
        注意：这里简化处理，实际可以使用 LLM 生成更好的摘要
        """
        session_info = self._active_sessions.get(session_id)
        
        if session_info and session_info.get('summary'):
            return session_info['summary']
        
        # 简单摘要：提取用户问题的关键词
        user_messages = [m['content'][:50] for m in messages if m['role'] == 'user']
        summary = "；".join(user_messages[:3]) + "..." if len(user_messages) > 3 else "；".join(user_messages)
        
        if session_info:
            session_info['summary'] = summary
        
        return summary
    
    async def clear_session(self, session_id: str):
        """清空会话"""
        await self.memory_store.clear_history(session_id)
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        logger.info(f"Session cleared: {session_id}")
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        await self.memory_store.delete_session(session_id)
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        logger.info(f"Session deleted: {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        return self._active_sessions.get(session_id)
    
    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return await self.memory_store.list_sessions(limit)
    
    async def cleanup(self, days: int = 7):
        """清理旧会话"""
        await self.memory_store.cleanup_old_sessions(days)
        
        # 清理活跃会话缓存
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        to_remove = [
            sid for sid, info in self._active_sessions.items()
            if datetime.fromisoformat(info['created_at']).timestamp() < cutoff
        ]
        for sid in to_remove:
            del self._active_sessions[sid]
        
        logger.info(f"Cleanup completed: removed {len(to_remove)} inactive sessions")


# 全局管理器实例
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    memory_store: Optional[SimpleMemoryStore] = None,
    max_context_length: int = 4000
) -> MemoryManager:
    """获取全局记忆管理器实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(memory_store, max_context_length)
    return _memory_manager
