from __future__ import annotations

from logger import get_logger
from typing import Optional
from pathlib import Path

from .local import LocalAgent
from .online import OpenAIAgent

logger = get_logger(__name__)


def _default_local_model():
    # 默认选择 ai/models/model_serving 下的第一个文件夹作为本地模型目录
    root = Path(__file__).resolve().parents[3]
    serving_dir = root / "ai" / "models" / "model_serving"
    if not serving_dir.exists() or not serving_dir.is_dir():
        raise FileNotFoundError(f"默认本地模型目录未找到：{serving_dir}")
    # 查找第一个子文件夹
    for item in serving_dir.iterdir():
        if item.is_dir():
            return str(item)
    raise FileNotFoundError(f"ai/models/model_serving 下未找到任何模型文件夹，请添加模型后重试。")


def get_agent(provider: str = "local", model: Optional[str] = None):
    """Return an agent implementation based on the provider string."""
    import time
    call_time = time.strftime('%H:%M:%S')
    logger.info(f"[{call_time}] === get_agent 被调用 ===")
    logger.info(f"[{call_time}] provider={provider}, model={model}")

    normalized = (provider or "").strip().lower()
    if normalized in {"online", "openai", "cloud"}:
        logger.info(f"[{call_time}] Using OpenAI agent provider (model=%s)", model or "gpt-3.5-turbo")
        return OpenAIAgent(model=model or "gpt-3.5-turbo")

    chosen_model = model or _default_local_model()
    logger.info(f"[{call_time}] Using Local agent provider (model=%s)", chosen_model)
    return LocalAgent(model_name=chosen_model)

