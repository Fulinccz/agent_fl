import time
from functools import wraps
from typing import Optional, Callable

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from logger import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """令牌桶限流算法"""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒产生令牌数
            capacity: 桶容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = False  # 简化版，实际用 threading.Lock

    def acquire(self, tokens: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """API 限流器"""

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
        key_func: Callable[[Request], str] = None
    ):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.key_func = key_func or self._default_key_func
        self._buckets: dict = {}

    @staticmethod
    def _default_key_func(request: Request) -> str:
        # 按 IP 限流
        client = request.client
        return client.host if client else "unknown"

    def _get_bucket(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                rate=self.requests_per_second,
                capacity=self.burst_size
            )
        return self._buckets[key]

    async def __call__(self, request: Request):
        key = self.key_func(request)
        bucket = self._get_bucket(key)

        if not bucket.acquire():
            logger.warning("Rate limit exceeded: %s", key)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": int(1 / self.requests_per_second) + 1
                }
            )


# 默认限流器实例
rate_limiter = RateLimiter(
    requests_per_second=10.0,
    burst_size=20
)
