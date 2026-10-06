"""Langfuse 全链路跟踪层（未配置时自动 no-op）

启用方式：在仓库根目录 .env 配置
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000   # 自托管默认地址；云版 https://cloud.langfuse.com

设计原则：
- 跟踪失败不影响业务：所有入口均降级为 no-op
- 层级自动正确：MAS 编排（agent）→ 意图/规划/执行/反思（span）→ LLM 调用（generation）
  依赖 OTEL context 在同线程同步调用中传播；provider 内部的线程池仅执行推理，
  span 在调用线程开/关，不受影响。
- langfuse v4（OTEL）API：start_as_current_observation + update_current_*。

使用方式：
    with start_span("mas-pipeline", as_type="agent", input=user_input):
        ...
        update_current_span(output={...})      # 补充当前 span 输出
        update_current_generation(output=text) # provider 内标记 LLM 输出
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from logger import get_logger

logger = get_logger(__name__)

_client: Optional[Any] = None
_inited = False
_lock = threading.Lock()

# 环境标识（production/staging/development），Langfuse environment 属性
_RUNTIME_ENV = os.getenv("FULIN_ENV", "development")


def get_langfuse() -> Optional[Any]:
    """获取 Langfuse 客户端单例；未配置或初始化失败时返回 None（no-op）"""
    global _client, _inited
    if _inited:
        return _client
    with _lock:
        if _inited:
            return _client
        _inited = True

        # 防御：确保 .env 已加载（Langfuse 必须在环境变量就绪后初始化）
        try:
            from dotenv import load_dotenv
            from services.app_config import ENV_FILE
            load_dotenv(ENV_FILE, override=False)
        except Exception:
            pass

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not public_key or not secret_key:
            logger.info("[Langfuse] 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，跟踪禁用")
            return None
        try:
            from langfuse import Langfuse
            _client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                environment=_RUNTIME_ENV,
            )
            logger.info("[Langfuse] 已启用: %s (env=%s)", host, _RUNTIME_ENV)
        except Exception as e:
            logger.warning("[Langfuse] 初始化失败，跟踪禁用: %s", e)
            _client = None
    return _client


def is_enabled() -> bool:
    return get_langfuse() is not None


def reset_for_tests():
    """重置单例（仅供测试）"""
    global _client, _inited
    with _lock:
        _client = None
        _inited = False


@contextmanager
def start_span(
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: Optional[dict] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
) -> Iterator[Optional[Any]]:
    """开启一个 observation 并设为当前上下文（嵌套调用自动形成层级）

    session_id/tags/environment 仅 trace 级生效（根 observation 上设置）。

    Args:
        name: observation 名称（动词优先、不含动态值，如 generate-response）
        as_type: span | generation | agent | tool | chain | retriever
        input/metadata: 初始输入与元数据
        session_id: 会话 ID（用于 Sessions 视图分组多轮对话）
        tags: 业务标签（创建时不可变）

    Yields:
        observation 对象（未启用跟踪时为 None）
    """
    lf = get_langfuse()
    if lf is None:
        yield None
        return

    try:
        kwargs = {"name": name, "as_type": as_type, "input": input}
        if metadata:
            kwargs["metadata"] = metadata
        if session_id:
            kwargs["session_id"] = session_id
        if tags:
            kwargs["tags"] = tags
        cm = lf.start_as_current_observation(**kwargs)
    except Exception as e:
        logger.warning("[Langfuse] 开启 span %s 失败: %s", name, e)
        yield None
        return

    with cm as obs:
        yield obs


def update_current_span(output: Any = None, metadata: Optional[dict] = None):
    """更新当前（span 类型）observation 的输出/元数据（未启用时 no-op）"""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.update_current_span(output=output, metadata=metadata)
    except Exception as e:
        logger.warning("[Langfuse] 更新 span 失败: %s", e)


def update_current_generation(output: Any = None, usage: Optional[dict] = None):
    """更新当前（generation 类型）observation 的输出/用量（未启用时 no-op）

    usage 形如 {"input": 12, "output": 34}（单位默认 TOKENS）
    """
    lf = get_langfuse()
    if lf is None:
        return
    try:
        kwargs = {"output": output}
        if usage is not None:
            kwargs["usage_details"] = usage
        lf.update_current_generation(**kwargs)
    except Exception as e:
        logger.warning("[Langfuse] 更新 generation 失败: %s", e)


def get_trace_url() -> Optional[str]:
    """获取当前 trace 的 Langfuse URL（未启用时 None，便于日志排查）"""
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        return lf.get_trace_url()
    except Exception:
        return None


def flush():
    """立即上报所有缓冲的跟踪数据（应用关闭时调用，长驻服务依赖批量导出）"""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as e:
        logger.warning("[Langfuse] flush 失败: %s", e)
