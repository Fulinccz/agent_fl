import json
import pickle
import hashlib
from functools import wraps
from typing import Optional, Any, Callable
import os

import redis
from redis.exceptions import RedisError

from logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Redis 缓存客户端（Cache-Aside 模式）"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        default_ttl: int = 300,  # 默认5分钟
        key_prefix: str = "cache:"
    ):
        from services.config import AppSettings
        config = AppSettings.load()
        self.host = host or config.redis_host
        self.port = port or config.redis_port
        self.db = db or config.redis_cache_db
        self.password = password or config.redis_password
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._client = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )
        return self._client

    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            data = self.client.get(self._make_key(key))
            if data:
                return pickle.loads(data.encode())
            return None
        except RedisError as e:
            logger.warning("Cache get failed: %s", e)
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        nx: bool = False
    ) -> bool:
        """设置缓存"""
        try:
            serialized = pickle.dumps(value)
            return self.client.set(
                self._make_key(key),
                serialized,
                ex=ttl or self.default_ttl,
                nx=nx
            )
        except RedisError as e:
            logger.warning("Cache set failed: %s", e)
            return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            return self.client.delete(self._make_key(key)) > 0
        except RedisError as e:
            logger.warning("Cache delete failed: %s", e)
            return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            return self.client.exists(self._make_key(key)) > 0
        except RedisError as e:
            logger.warning("Cache exists failed: %s", e)
            return False

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: int = None
    ) -> Any:
        """Cache-Aside 模式：先读缓存，不存在则调用 factory 写入"""
        value = self.get(key)
        if value is not None:
            logger.debug("Cache hit: %s", key)
            return value

        logger.debug("Cache miss: %s", key)
        value = factory()
        if value is not None:
            self.set(key, value, ttl)
        return value

    def invalidate_pattern(self, pattern: str) -> int:
        """按模式删除缓存"""
        try:
            keys = self.client.keys(self._make_key(pattern))
            if keys:
                return self.client.delete(*keys)
            return 0
        except RedisError as e:
            logger.warning("Cache invalidate failed: %s", e)
            return 0


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Callable = None
):
    """缓存装饰器"""
    cache = RedisCache(default_ttl=ttl, key_prefix=key_prefix)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # 默认 key: function_name:args_hash
                sig = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(sig.encode()).hexdigest()

            return cache.get_or_set(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl=ttl
            )
        return wrapper
    return decorator


# 全局缓存实例
cache = RedisCache()
