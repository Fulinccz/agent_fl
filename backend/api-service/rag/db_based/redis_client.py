import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from redis.exceptions import RedisError
from typing import List, Dict, Any, Optional


class RAGRedisClient:
    """RAG 模块专用的 Redis 客户端"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        key_prefix: str = "resume:",
        set_key: str = "resumes:ids"
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD", "") or None
        self.key_prefix = key_prefix
        self.set_key = set_key
        self._client = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
        return self._client

    def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """获取单条简历"""
        try:
            key = f"{self.key_prefix}{resume_id}"
            data = self.client.hgetall(key)
            return dict(data) if data else None
        except RedisError as e:
            print(f"[ERROR] Redis get failed: {e}")
            return None

    def get_all_resume_ids(self) -> List[str]:
        """获取所有简历 ID"""
        try:
            return list(self.client.smembers(self.set_key))
        except RedisError as e:
            print(f"[ERROR] Redis get ids failed: {e}")
            return []

    def get_resumes_batch(self, resume_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取简历"""
        results = []
        for rid in resume_ids:
            resume = self.get_resume(rid)
            if resume:
                results.append(resume)
        return results

    def get_all_resumes(self, limit: int = None) -> List[Dict[str, Any]]:
        """获取所有简历（分页）"""
        ids = self.get_all_resume_ids()
        if limit:
            ids = ids[:limit]
        return self.get_resumes_batch(ids)

    def get_resume_text(self, resume: Dict[str, Any]) -> str:
        """将简历拼接为文本用于向量化"""
        parts = []
        if resume.get("target_position"):
            parts.append(f"意向岗位: {resume['target_position']}")
        if resume.get("degree"):
            parts.append(f"学历: {resume['degree']}")
        if resume.get("university_type"):
            parts.append(f"院校: {resume['university_type']}")
        if resume.get("work_description"):
            parts.append(f"工作经历: {resume['work_description']}")
        if resume.get("project_description"):
            parts.append(f"项目经历: {resume['project_description']}")
        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        try:
            ids = self.get_all_resume_ids()
            return {
                "total_resumes": len(ids),
                "key_prefix": self.key_prefix,
                "set_key": self.set_key
            }
        except RedisError as e:
            return {"error": str(e)}
