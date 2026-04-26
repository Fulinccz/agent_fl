"""
Local Agent - 向后兼容的适配器类
"""

from __future__ import annotations
from typing import Dict, Any, Generator, Optional, List
from .provider import LocalProvider


class LocalAgent:
    """
    Local Agent - 向后兼容的适配器类
    
    此类是为了与旧代码兼容而保留
    实际功能由 LocalProvider 实现
    """

    def __init__(self, model_name: str = "model_serving"):
        """Initialize local agent."""
        self.provider = LocalProvider(model_name=model_name)

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text using local model."""
        return self.provider.generate(prompt, images, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate text with streaming."""
        yield from self.provider.generate_stream(prompt, images, **kwargs)

    def stop_generation(self):
        """Stop current generation."""
        self.provider.stop_generation()
