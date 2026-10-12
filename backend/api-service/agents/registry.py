"""
Agent Registry

本地链路（GGUF）与在线链路（DeepSeek）入口分离：
- get_provider("local")    → _get_local_provider（GGUF，按模型路径缓存，延迟 import）
- get_provider("deepseek") → _get_cloud_provider（DeepSeek，延迟 import）

两条链路物理隔离（providers/local/ vs providers/cloud/），改一条不影响另一条。
顶部不再硬 import 在线 Provider 类，本地链路启动不加载在线 SDK。
"""

from __future__ import annotations

import threading

from logger import get_logger
from typing import Optional, Dict, Any

logger = get_logger(__name__)


class AgentRegistry:

    _instance = None
    _providers: Dict[str, Any] = {}
    # 本地 Provider 实例缓存：key=模型路径（加载代价高，必须复用）
    _local_provider_cache: Dict[str, Any] = {}
    _local_cache_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._default_model = None
        logger.info("AgentRegistry 初始化完成")

    def get_provider(
        self,
        provider: str = "local",
        model: Optional[str] = None,
        **kwargs
    ):
        """
        获取模型提供者实例（兼容入口，按 provider 分发到本地/在线链路）

        Args:
            provider: 提供者类型 ('local' | 'deepseek')；'local'/缺省走本地 GGUF
            model: 模型名称或路径（local 时忽略，统一用 config 指定的 GGUF）
            **kwargs: 额外参数（如 api_key）

        Returns:
            Provider 实例 (GGUFProvider / DeepSeekProvider)
        """
        import time
        call_time = time.strftime('%H:%M:%S')
        normalized = (provider or "").strip().lower()

        logger.info(f"[{call_time}] === get_provider 被调用 ===")
        logger.info(f"[{call_time}] provider={normalized}, model={model}")

        if normalized == "deepseek":
            return self._get_cloud_provider(normalized, model, **kwargs)

        # 其余一律走本地链路（OnlineProvider/openai/cloud 已随死代码删除）
        return self._get_local_provider(model, **kwargs)

    # ------------------------------------------------------------------
    # 本地链路（GGUF + llama.cpp）
    # ------------------------------------------------------------------

    def _get_local_provider(self, model: Optional[str] = None, **kwargs):
        """本地推理统一走 GGUF（llama.cpp Q4 量化，内存约 2.6GB）。

        加载代价高（首次解压二进制 + 模型入 RAM），按 gguf_path 缓存实例复用。
        """
        import time
        call_time = time.strftime('%H:%M:%S')

        gguf_path = self._get_gguf_path()
        cache_key = f"gguf::{gguf_path}"
        cached = self._local_provider_cache.get(cache_key)
        if cached is not None:
            logger.info(f"[{call_time}] Using cached GGUF provider ({gguf_path})")
            return cached

        from .providers.local.gguf_provider import GGUFProvider
        logger.info(f"[{call_time}] Using GGUF provider ({gguf_path})")
        provider_instance = GGUFProvider(model_name=gguf_path)
        with self._local_cache_lock:
            self._local_provider_cache[cache_key] = provider_instance
        return provider_instance

    # ------------------------------------------------------------------
    # 在线链路（DeepSeek，OpenAI 兼容协议）
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cloud_provider(normalized: str, model: Optional[str] = None, **kwargs):
        """在线 Provider（延迟 import，避免本地链路启动加载在线 SDK）"""
        import time
        call_time = time.strftime('%H:%M:%S')

        if normalized == "deepseek":
            logger.info(f"[{call_time}] Using DeepSeek provider (model={model or 'config default'})")
            from .providers.cloud.deepseek import DeepSeekProvider
            return DeepSeekProvider(
                api_key=kwargs.get('api_key'),
                model=model,
            )

        # 理论上不会到达（get_provider 已分发），兜底抛错
        raise ValueError(f"未知的在线 provider: {normalized}")

    # ------------------------------------------------------------------
    # 本地链路辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gguf_path() -> str:
        """读取 config models.local.gguf_path（ai/models 相对路径），无配置用默认"""
        try:
            from services.app_config import load_app_config
            return (load_app_config().models.local.gguf_path or "").strip() \
                or "gguf/Qwen3.5-4B-Q4_K_M.gguf"
        except Exception:
            return "gguf/Qwen3.5-4B-Q4_K_M.gguf"

    @classmethod
    def clear_provider_cache(cls) -> None:
        """清空 Provider 缓存（测试或热更换模型时使用）"""
        with cls._local_cache_lock:
            cls._local_provider_cache.clear()


# 向后兼容：保留原有的 get_agent 函数
def get_agent(provider: str = "local", model: Optional[str] = None):
    """
    获取 Agent 实例（向后兼容接口）

    内部调用 AgentRegistry.get_provider 做本地/在线分发。

    Args:
        provider: 提供者类型（'local' | 'deepseek'）
        model: 模型名称

    Returns:
        Provider 实例
    """
    registry = AgentRegistry()
    return registry.get_provider(provider=provider, model=model)


__all__ = [
    'AgentRegistry',
    'get_agent',
]
