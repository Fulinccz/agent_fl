from __future__ import annotations

from typing import Optional

from .local import LocalAgent
from .online import OpenAIAgent


def get_agent(provider: str = "local", model: Optional[str] = None):
    """Return an agent implementation based on the provider string."""

    normalized = (provider or "").strip().lower()
    if normalized in {"online", "openai", "cloud"}:
        return OpenAIAgent(model=model or "gpt-3.5-turbo")

    # 支持通过model参数指定qwen3.5模型
    return LocalAgent(model_name=model or "Qwen/Qwen3.5-4B")
