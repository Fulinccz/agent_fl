from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text given a prompt."""
        raise NotImplementedError
