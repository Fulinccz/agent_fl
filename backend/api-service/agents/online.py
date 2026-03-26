from __future__ import annotations

from logger import get_logger
import os
from typing import Optional

from .base import BaseAgent

logger = get_logger(__name__)


class OpenAIAgent(BaseAgent):
    """A simple OpenAI API adapter."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        try:
            import openai
        except ImportError:  # pragma: no cover
            openai = None

        self._openai = openai
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if self._openai and self.api_key:
            self._openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        if self._openai is None:
            logger.error("OpenAI SDK 未安装，无法生成文本")
            return "[openai-sdk-not-installed] " + prompt

        if not self.api_key:
            logger.error("OpenAI API Key 未设置")
            return "[openai-api-key-not-set] " + prompt

        resp = self._openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content
