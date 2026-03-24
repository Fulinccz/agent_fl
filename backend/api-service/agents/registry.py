from __future__ import annotations

from typing import Optional
from pathlib import Path

from .local import LocalAgent
from .online import OpenAIAgent


def _default_local_model():
    # 仅支持仓库中 ai/models 下的本地模型目录
    root = Path(__file__).resolve().parents[3]
    local_model = root / "ai" / "models" / "Qwen3___5-4B"
    if local_model.exists():
        return str(local_model)
    raise FileNotFoundError(
        f"默认本地模型未找到：{local_model}。请在 ai/models 下放置模型或显式指定 model 参数。"
    )


def get_agent(provider: str = "local", model: Optional[str] = None):
    """Return an agent implementation based on the provider string."""

    normalized = (provider or "").strip().lower()
    if normalized in {"online", "openai", "cloud"}:
        return OpenAIAgent(model=model or "gpt-3.5-turbo")

    chosen_model = model or _default_local_model()
    return LocalAgent(model_name=chosen_model)

