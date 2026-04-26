"""
简单内存存储 - 用于快速测试

使用内存字典存储对话历史，适合测试和轻量级场景。
生产环境建议使用 SQLiteMemoryStore。
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationMemory:
    """单条对话记忆"""
    role: str
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )


class SimpleMemoryStore:
    """
    简单内存存储
    
    使用字典存储对话历史，数据在内存中，重启后丢失。
    适合测试和演示。
    """
    
    def __init__(self, max_history: int = 10):
        """
        初始化存储
        
        Args:
            max_history: 每个会话最多保留的对话轮数
        """
        self.max_history = max_history
        self._storage: Dict[str, List[ConversationMemory]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"SimpleMemoryStore initialized: max_history={max_history}")
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加对话消息"""
        if session_id not in self._storage:
            self._storage[session_id] = []
            self._sessions[session_id] = {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
        
        memory = ConversationMemory(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self._storage[session_id].append(memory)
        self._sessions[session_id]['updated_at'] = datetime.now().isoformat()
        
        # 限制历史长度
        if len(self._storage[session_id]) > self.max_history:
            self._storage[session_id] = self._storage[session_id][-self.max_history:]
        
        logger.debug(f"Message added to session {session_id}: {role}")
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMemory]:
        """获取对话历史"""
        history = self._storage.get(session_id, [])
        
        if limit:
            history = history[-limit:]
        
        return history
    
    async def get_history_as_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """获取对话历史，格式化为消息格式"""
        history = await self.get_history(session_id, limit)
        return [{'role': m.role, 'content': m.content} for m in history]
    
    async def clear_history(self, session_id: str):
        """清空指定会话的历史"""
        if session_id in self._storage:
            del self._storage[session_id]
        if session_id in self._sessions:
            del self._sessions[session_id]
        logger.info(f"History cleared for session {session_id}")
    
    async def delete_session(self, session_id: str):
        """删除整个会话"""
        await self.clear_history(session_id)
    
    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有会话"""
        sessions = []
        for session_id, info in self._sessions.items():
            sessions.append({
                'session_id': session_id,
                'created_at': info['created_at'],
                'updated_at': info['updated_at'],
                'metadata': info.get('metadata', {})
            })
        
        # 按更新时间排序
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        return sessions[:limit]
    
    async def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
        total_messages = sum(len(h) for h in self._storage.values())
        return {
            'total_sessions': len(self._storage),
            'total_messages': total_messages
        }


# 全局存储实例
_memory_store: Optional[SimpleMemoryStore] = None


def get_simple_memory_store(max_history: int = 10) -> SimpleMemoryStore:
    """获取全局简单记忆存储实例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = SimpleMemoryStore(max_history)
    return _memory_store
