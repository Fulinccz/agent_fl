"""
SQLite 记忆存储模块
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import aiosqlite
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationMemory:
    """单条对话记忆"""
    role: str  # 'user' 或 'assistant'
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


class SQLiteMemoryStore:
    """
    SQLite 对话记忆存储
    
    使用 SQLite 本地数据库存储对话历史，支持：
    - 多会话隔离
    - 记忆长度限制（默认保留最近 10 轮）
    - 自动清理过期会话
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        max_history: int = 10,  # 最多保留 10 轮对话（5轮问答）
        max_sessions: int = 1000  # 最多保留 1000 个会话
    ):
        """
        初始化存储
        
        Args:
            db_path: 数据库文件路径，默认在 data/memory.db
            max_history: 每个会话最多保留的对话轮数
            max_sessions: 最多保留的会话数量
        """
        self.db_path = db_path or os.path.join(os.getcwd(), "data", "memory.db")
        self.max_history = max_history
        self.max_sessions = max_sessions
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        logger.info(f"SQLiteMemoryStore initialized: {self.db_path}, max_history={max_history}")
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 对话历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session 
                ON conversations(session_id, timestamp)
            """)
            
            conn.commit()
            logger.debug("Database tables initialized")
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加对话消息
        
        Args:
            session_id: 会话 ID
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            metadata: 可选的元数据
        """
        async with aiosqlite.connect(self.db_path) as db:
            # 确保会话存在
            await db.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, updated_at, metadata)
                VALUES (?, CURRENT_TIMESTAMP, ?)
                """,
                (session_id, json.dumps(metadata) if metadata else None)
            )
            
            # 插入消息
            await db.execute(
                """
                INSERT INTO conversations (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, json.dumps(metadata) if metadata else None)
            )
            
            await db.commit()
            
            # 清理旧消息，只保留最近的 max_history 条
            await self._trim_history(db, session_id)
            
            logger.debug(f"Message added to session {session_id}: {role}")
    
    async def _trim_history(self, db: aiosqlite.Connection, session_id: str):
        """修剪历史记录，只保留最近的 max_history 条"""
        cursor = await db.execute(
            """
            SELECT id FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT -1 OFFSET ?
            """,
            (session_id, self.max_history)
        )
        
        old_ids = await cursor.fetchall()
        if old_ids:
            ids_to_delete = [row[0] for row in old_ids]
            placeholders = ','.join('?' * len(ids_to_delete))
            await db.execute(
                f"DELETE FROM conversations WHERE id IN ({placeholders})",
                ids_to_delete
            )
            await db.commit()
            logger.debug(f"Trimmed {len(old_ids)} old messages from session {session_id}")
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMemory]:
        """
        获取对话历史
        
        Args:
            session_id: 会话 ID
            limit: 限制返回的消息数量
            
        Returns:
            对话历史列表
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = """
                SELECT role, content, timestamp, metadata 
                FROM conversations 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """
            params = [session_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                memory = ConversationMemory(
                    role=row[0],
                    content=row[1],
                    timestamp=row[2],
                    metadata=json.loads(row[3]) if row[3] else None
                )
                history.append(memory)
            
            return history
    
    async def get_history_as_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        获取对话历史，格式化为 LangChain 消息格式
        
        Returns:
            [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...]
        """
        history = await self.get_history(session_id, limit)
        return [{'role': m.role, 'content': m.content} for m in history]
    
    async def clear_history(self, session_id: str):
        """清空指定会话的历史"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,)
            )
            await db.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            logger.info(f"History cleared for session {session_id}")
    
    async def delete_session(self, session_id: str):
        """删除整个会话"""
        await self.clear_history(session_id)
    
    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有会话"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT session_id, created_at, updated_at, metadata 
                FROM sessions 
                ORDER BY updated_at DESC 
                LIMIT ?
                """,
                (limit,)
            )
            rows = await cursor.fetchall()
            
            sessions = []
            for row in rows:
                sessions.append({
                    'session_id': row[0],
                    'created_at': row[1],
                    'updated_at': row[2],
                    'metadata': json.loads(row[3]) if row[3] else None
                })
            
            return sessions
    
    async def cleanup_old_sessions(self, days: int = 7):
        """
        清理长时间未活动的会话
        
        Args:
            days: 清理超过 N 天未活动的会话
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT session_id FROM sessions 
                WHERE updated_at < datetime('now', '-{} days')
                """.format(days)
            )
            rows = await cursor.fetchall()
            
            for row in rows:
                await self.delete_session(row[0])
            
            if rows:
                logger.info(f"Cleaned up {len(rows)} old sessions")
    
    async def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM sessions")
            session_count = (await cursor.fetchone())[0]
            
            cursor = await db.execute("SELECT COUNT(*) FROM conversations")
            message_count = (await cursor.fetchone())[0]
            
            return {
                'total_sessions': session_count,
                'total_messages': message_count
            }


# 全局存储实例
_memory_store: Optional[SQLiteMemoryStore] = None


def get_memory_store(
    db_path: Optional[str] = None,
    max_history: int = 10
) -> SQLiteMemoryStore:
    """获取全局记忆存储实例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = SQLiteMemoryStore(db_path, max_history)
    return _memory_store
