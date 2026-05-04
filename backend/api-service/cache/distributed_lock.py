"""
分布式锁（基于 Redis Redlock 算法简化版）

适用场景：
1. 定时任务防重
2. 资源重建防并发
3. 消息幂等处理

用法：
    async with distributed_lock("job:crawl:java", timeout=300):
        await crawl_jobs("java")
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis
from redis.exceptions import RedisError

from logger import get_logger

logger = get_logger(__name__)


class DistributedLock:
    """Redis 分布式锁"""

    def __init__(
        self,
        redis_client: redis.Redis = None,
        default_timeout: int = 60,
        block_timeout: int = 10
    ):
        self.redis = redis_client
        self.default_timeout = default_timeout
        self.block_timeout = block_timeout

    def _get_redis(self) -> redis.Redis:
        if self.redis is None:
            import os
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_LOCK_DB", "2")),  # 专用 DB2
                decode_responses=True,
                socket_connect_timeout=5,
            )
        return self.redis

    def acquire(
        self,
        lock_key: str,
        timeout: int = None,
        block: bool = True
    ) -> Optional[str]:
        """
        获取锁

        Args:
            lock_key: 锁标识
            timeout: 锁超时时间（秒）
            block: 是否阻塞等待

        Returns:
            token: 成功返回 token，失败返回 None
        """
        redis_client = self._get_redis()
        timeout = timeout or self.default_timeout
        token = str(uuid.uuid4())
        full_key = f"lock:{lock_key}"

        start_time = time.time()

        while True:
            try:
                # NX: 只有 key 不存在才设置
                # EX: 设置过期时间
                acquired = redis_client.set(
                    full_key,
                    token,
                    nx=True,
                    ex=timeout
                )

                if acquired:
                    logger.debug("Lock acquired: %s", lock_key)
                    return token

                if not block:
                    return None

                # 阻塞等待，带超时
                if time.time() - start_time > self.block_timeout:
                    logger.warning("Lock acquire timeout: %s", lock_key)
                    return None

                time.sleep(0.1)

            except RedisError as e:
                logger.error("Lock acquire failed: %s", e)
                return None

    def release(self, lock_key: str, token: str) -> bool:
        """
        释放锁（安全释放，只有持有者可释放）

        Args:
            lock_key: 锁标识
            token: 获取锁时返回的 token

        Returns:
            是否释放成功
        """
        redis_client = self._get_redis()
        full_key = f"lock:{lock_key}"

        # Lua 脚本保证原子性：只有 token 匹配才删除
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = redis_client.eval(lua_script, 1, full_key, token)
            if result:
                logger.debug("Lock released: %s", lock_key)
                return True
            else:
                logger.warning("Lock release failed (not owner): %s", lock_key)
                return False
        except RedisError as e:
            logger.error("Lock release failed: %s", e)
            return False

    def extend(self, lock_key: str, token: str, additional_time: int) -> bool:
        """延长锁时间（业务执行时间超预期时）"""
        redis_client = self._get_redis()
        full_key = f"lock:{lock_key}"

        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        try:
            result = redis_client.eval(
                lua_script, 1, full_key, token, str(additional_time)
            )
            return bool(result)
        except RedisError:
            return False


# 全局锁实例
lock = DistributedLock()


@asynccontextmanager
async def distributed_lock(
    lock_key: str,
    timeout: int = 60,
    block: bool = True
):
    """
    异步上下文管理器用法

    Example:
        async with distributed_lock("task:daily_crawl", timeout=300):
            await do_crawl()
    """
    token = lock.acquire(lock_key, timeout=timeout, block=block)
    if token is None:
        raise LockAcquireError(f"Failed to acquire lock: {lock_key}")

    try:
        yield token
    finally:
        lock.release(lock_key, token)


class LockAcquireError(Exception):
    """锁获取失败异常"""
    pass
