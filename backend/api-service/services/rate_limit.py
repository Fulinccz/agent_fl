"""简易内存限流（滑动窗口计数）

Demo 级实现：不引入 Redis，进程内按 IP 统计请求次数；
多实例部署或生产环境应替换为 Redis 实现。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Request

# 端点 → (最大次数, 窗口秒数)
DEFAULT_LIMIT = (10, 60)
LIMITS: dict[str, tuple[int, int]] = {
    "/api/resume/optimize/stream": DEFAULT_LIMIT,
    "/api/agent/upload_stream": DEFAULT_LIMIT,
    "/api/resume/export-docx": (20, 60),
}

# 全局兜底（所有端点共用一套宽松配额；与端点级严格限制分桶计数，互不叠加）
GLOBAL_LIMIT = (120, 60)

_buckets: dict[str, deque] = defaultdict(deque)
_global_buckets: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def check_rate_limit(request: Request, path: str | None = None) -> tuple[bool, int, int]:
    """检查限流。返回 (allowed, retry_after_seconds, remaining)

    超限时不移除记录（自然过期），调用方返回 429。
    """
    limit, window = LIMITS.get(path or request.url.path, DEFAULT_LIMIT)
    key = f"{_client_ip(request)}:{path or request.url.path}"
    now = time.monotonic()

    with _lock:
        bucket = _buckets[key]
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + window - now) + 1)
            return False, retry_after, 0
        bucket.append(now)
        return True, 0, limit - len(bucket)


def check_global_rate_limit(request: Request) -> tuple[bool, int, int]:
    """全局限流（main.py 中间件调用，按 IP 宽松兜底）。

    与 check_rate_limit 同一套滑动窗口算法、独立桶。返回 (allowed, retry_after_seconds, remaining)。
    """
    key = _client_ip(request)
    limit, window = GLOBAL_LIMIT
    now = time.monotonic()

    with _lock:
        bucket = _global_buckets[key]
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + window - now) + 1)
            return False, retry_after, 0
        bucket.append(now)
        return True, 0, limit - len(bucket)


def reset_rate_limits():
    """清空计数（测试用）"""
    with _lock:
        _buckets.clear()
        _global_buckets.clear()
