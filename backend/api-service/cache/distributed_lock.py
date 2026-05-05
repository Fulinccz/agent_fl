"""
分布式锁
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from redis.exceptions import RedisError

from logger import get_logger

logger = get_logger(__name__)


class DistributedLock:
    def __init__(
        self,
        redis_client: redis.Redis = None,
        default_timeout: int = 60,
        block_timeout: int = 10
    ):
        self.redis = redis_client
        self.default_timeout = default_timeout
        self.block_timeout = block_timeout

    async def _get_redis(self) -> redis.Redis:
        if self.redis is None:
            from services.config import AppSettings
            config = AppSettings.load()
            self.redis = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                password=config.redis_password,
                db=config.redis_lock_db,
                decode_responses=True,
                socket_connect_timeout=5,
            )
        return self.redis

    async def acquire(
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
        redis_client = await self._get_redis()
        timeout = timeout or self.default_timeout
        token = str(uuid.uuid4())
        full_key = f"lock:{lock_key}"

        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                acquired = await redis_client.set(
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

                if asyncio.get_event_loop().time() - start_time > self.block_timeout:
                    logger.warning("Lock acquire timeout: %s", lock_key)
                    return None

                await asyncio.sleep(0.1)

            except RedisError as e:
                logger.error("Lock acquire failed: %s", e)
                return None

    async def release(self, lock_key: str, token: str) -> bool:
        """
        释放锁（安全释放，只有持有者可释放）

        Args:
            lock_key: 锁标识
            token: 获取锁时返回的 token

        Returns:
            是否释放成功
        """
        redis_client = await self._get_redis()
        full_key = f"lock:{lock_key}"

        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await redis_client.eval(lua_script, 1, full_key, token)
            if result:
                logger.debug("Lock released: %s", lock_key)
                return True
            else:
                logger.warning("Lock release failed (not owner): %s", lock_key)
                return False
        except RedisError as e:
            logger.error("Lock release failed: %s", e)
            return False

    async def extend(self, lock_key: str, token: str, additional_time: int) -> bool:
        """延长锁时间（业务执行时间超预期时）"""
        redis_client = await self._get_redis()
        full_key = f"lock:{lock_key}"

        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        try:
            result = await redis_client.eval(
                lua_script, 1, full_key, token, str(additional_time)
            )
            return bool(result)
        except RedisError:
            return False


# 全局锁实例
_lock = DistributedLock()


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
    token = await _lock.acquire(lock_key, timeout=timeout, block=block)
    if token is None:
        raise LockAcquireError(f"Failed to acquire lock: {lock_key}")

    try:
        yield token
    finally:
        await _lock.release(lock_key, token)


class LockAcquireError(Exception):
    """锁获取失败异常"""
    pass
