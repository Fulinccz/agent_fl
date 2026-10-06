"""统一的 LLM 共享单例管理

合并自 langgraph.resume_agents.workflow 与 mas.llm 的两套实现，
消除双向依赖，成为全系统唯一的 LLM 访问入口。

本地模型加载代价高（4bit 量化 + GPU 显存），全系统共用同一个 Provider 实例；
在线 Provider（DeepSeek 等）创建代价低，按 (provider, model) 缓存即可。

路由规则：
    get_shared_llm()                          → 默认本地模型单例（预加载生效）
    get_shared_llm(provider="local", model=x) → 指定本地模型（缓存复用）
    get_shared_llm(provider="deepseek")       → DeepSeekProvider（成本开关，缓存复用）
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

from logger import get_logger

logger = get_logger(__name__)

# ---- 本地默认单例（带锁，预加载线程安全） ----
_shared_llm_instance: Optional[Any] = None
_model_loading = False
_model_loaded = False
_llm_lock = threading.Lock()

# ---- 非默认 Provider 缓存（按 provider+model） ----
_extra_llms: Dict[Tuple[str, Optional[str]], Any] = {}


def _provider_enabled(provider: str) -> bool:
    """在线 Provider 成本开关（config.yaml [models.<provider>] enabled，默认关闭）"""
    try:
        from services.app_config import load_app_config
        cfg = load_app_config().models
        section = getattr(cfg, provider, None)
        return bool(getattr(section, "enabled", False))
    except Exception as e:
        logger.warning(f"[agents.llm] 读取 provider 开关失败: {e}")
        return False


def preload_model() -> None:
    """后台预加载本地模型（在 main.py lifespan 中调用）"""
    global _model_loading
    if _model_loaded or _model_loading:
        return
    _model_loading = True
    try:
        logger.info("[agents.llm] 后台预加载 LLM 模型...")
        get_shared_llm()
        logger.info("[agents.llm] LLM 模型预加载完成")
    except Exception as e:
        logger.error(f"[agents.llm] 模型预加载失败: {e}")
    finally:
        _model_loading = False


def get_shared_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Any:
    """获取共享 LLM Provider。

    Args:
        provider: 提供者名称（local/deepseek/openai...），None 时使用默认本地单例
        model: 模型名称或路径，None 时使用该 provider 的默认模型

    Returns:
        Provider 实例
    """
    global _shared_llm_instance, _model_loaded

    normalized = (provider or "").strip().lower()

    # ---- 默认路径：本地单例 ----
    if not normalized or normalized == "local":
        if model is None:
            if _shared_llm_instance is None:
                with _llm_lock:
                    if _shared_llm_instance is None:
                        from agents.registry import get_agent
                        logger.info("[agents.llm] 初始化全局共享 LLM 实例")
                        _shared_llm_instance = get_agent(provider="local")
                        _model_loaded = True
            return _shared_llm_instance
        key: Tuple[str, Optional[str]] = ("local", model)
    else:
        # ---- 在线 Provider：成本开关，未启用回落本地 ----
        if not _provider_enabled(normalized):
            logger.warning(
                "[agents.llm] provider=%s 未启用（成本控制），回落本地模型", normalized
            )
            return get_shared_llm(provider="local", model=None)
        key = (normalized, model)

    # ---- 非默认 Provider 缓存 ----
    if key not in _extra_llms:
        from agents.registry import get_agent
        logger.info(f"[agents.llm] 初始化共享 LLM 实例: provider={key[0]}, model={key[1]}")
        _extra_llms[key] = get_agent(provider=key[0], model=key[1])
    return _extra_llms[key]


__all__ = ["get_shared_llm", "preload_model"]
