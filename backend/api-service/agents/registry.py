"""
Agent Registry
"""

from __future__ import annotations

from logger import get_logger
from typing import Optional, Dict, Any
from pathlib import Path

# 导入新的架构组件
from .providers.local import LocalProvider
from .providers.online import OnlineProvider, OpenAIAgent
from .providers.deepseek import DeepSeekProvider

logger = get_logger(__name__)


class AgentRegistry:

    _instance = None
    _providers: Dict[str, Any] = {}

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
    
    @staticmethod
    def _get_default_local_model() -> str:
        """获取默认的本地模型路径"""
        root = Path(__file__).resolve().parents[3]
        serving_dir = root / "ai" / "models" / "model_serving"
        
        if not serving_dir.exists() or not serving_dir.is_dir():
            raise FileNotFoundError(f"默认本地模型目录未找到：{serving_dir}")
        
        for item in serving_dir.iterdir():
            if item.is_dir():
                return str(item)
        
        raise FileNotFoundError(
            f"ai/models/model_serving 下未找到任何模型文件夹，请添加模型后重试。"
        )
    
    def get_provider(
        self,
        provider: str = "local",
        model: Optional[str] = None,
        **kwargs
    ):
        """
        获取模型提供者实例

        Args:
            provider: 提供者类型 ('local', 'deepseek', 'online', 'openai', 'cloud')
            model: 模型名称或路径
            **kwargs: 额外参数（如 api_key）

        Returns:
            Provider 实例 (LocalProvider / DeepSeekProvider / OnlineProvider)
        """
        import time
        call_time = time.strftime('%H:%M:%S')
        normalized = (provider or "").strip().lower()

        logger.info(f"[{call_time}] === get_provider 被调用 ===")
        logger.info(f"[{call_time}] provider={normalized}, model={model}")

        if normalized == "deepseek":
            logger.info(f"[{call_time}] Using DeepSeek provider (model={model or 'config default'})")
            return DeepSeekProvider(
                api_key=kwargs.get('api_key'),
                model=model,
            )

        if normalized in {"online", "openai", "cloud"}:
            logger.info(f"[{call_time}] Using Online provider (model={model or 'gpt-3.5-turbo'})")

            api_key = kwargs.get('api_key')
            return OnlineProvider(api_key=api_key, model=model or "gpt-3.5-turbo")
        
        chosen_model = model or self._get_default_local_model()
        logger.info(f"[{call_time}] Using Local provider (model={chosen_model})")

        return LocalProvider(model_name=chosen_model)


# 向后兼容：保留原有的 get_agent 函数
def get_agent(provider: str = "local", model: Optional[str] = None):
    """
    获取 Agent 实例（向后兼容接口）
    
    此函数保持与旧代码完全兼容，内部调用新的 AgentRegistry
    
    Args:
        provider: 提供者类型
        model: 模型名称
        
    Returns:
        Provider 实例
    """
    registry = AgentRegistry()
    return registry.get_provider(provider=provider, model=model)


# 便捷导出
__all__ = [
    'AgentRegistry',
    'get_agent',
    'LocalProvider',
    'OnlineProvider',
    'OpenAIAgent',
    'DeepSeekProvider',
]
