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
    "/api/auth/login": (5, 60),
    "/api/resume/optimize/stream": DEFAULT_LIMIT,
    "/api/agent/upload_stream": DEFAULT_LIMIT,
    "/api/resume/export-docx": (20, 60),
}

# 全局兜底（所有端点共用一套宽松配额；与端点级严格限制分桶计数，互不叠加）
GLOBAL_LIMIT = (120, 60)

# 桶淘汰：超过此时长没有任何请求的 IP 桶直接移除，防止桶数量随访问 IP 无界增长
_EVICT_TTL_SECONDS = 300
_EVICT_INTERVAL_SECONDS = 60

_buckets: dict[str, deque] = defaultdict(deque)
_global_buckets: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()
_last_eviction = 0.0


def _client_ip(request: Request) -> str:
    """获取客户端 IP。

    部署链路为 nginx → Go 网关 → 本服务，边缘 nginx 会用
    X-Forwarded-For $proxy_add_x_forwarded_for 写入真实客户端，
    因此优先取 XFF 的第一个地址；直连场景回退到 TCP 对端地址。
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]  # 超长头不做为键，防止恶意内存放大
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _evict_if_needed(now: float) -> None:
    """惰性淘汰：每隔一小段时间清理一次过期桶（调用方需持有 _lock）"""
    global _last_eviction
    if now - _last_eviction < _EVICT_INTERVAL_SECONDS:
        return
    _last_eviction = now
    for buckets in (_buckets, _global_buckets):
        stale = [
            key for key, bucket in buckets.items()
            if not bucket or bucket[-1] < now - _EVICT_TTL_SECONDS
        ]
        for key in stale:
            del buckets[key]


def _check(buckets: dict[str, deque], key: str, limit: int, window: int,
           now: float) -> tuple[bool, int, int]:
    with _lock:
        _evict_if_needed(now)
        bucket = buckets[key]
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + window - now) + 1)
            return False, retry_after, 0
        bucket.append(now)
        return True, 0, limit - len(bucket)


def check_rate_limit(request: Request, path: str | None = None) -> tuple[bool, int, int]:
    """检查限流。返回 (allowed, retry_after_seconds, remaining)

    超限时不记录本次请求（自然过期），调用方返回 429。
    """
    route = path or request.url.path
    limit, window = LIMITS.get(route, DEFAULT_LIMIT)
    key = f"{_client_ip(request)}:{route}"
    return _check(_buckets, key, limit, window, time.monotonic())


def check_global_rate_limit(request: Request) -> tuple[bool, int, int]:
    """全局限流（main.py 中间件调用，按 IP 宽松兜底）。

    与 check_rate_limit 同一套滑动窗口算法、独立桶。返回 (allowed, retry_after_seconds, remaining)。
    """
    return _check(_global_buckets, _client_ip(request), *GLOBAL_LIMIT, time.monotonic())


def reset_rate_limits():
    """清空计数（测试用）"""
    global _last_eviction
    with _lock:
        _buckets.clear()
        _global_buckets.clear()
        _last_eviction = 0.0
