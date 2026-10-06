"""
Online Provider
"""

from __future__ import annotations

from logger import get_logger
import os
from typing import Optional, List, Dict, Generator, Any

from ..core.base_provider import BaseProvider

logger = get_logger(__name__)


class OnlineProvider(BaseProvider):
    """
    provider = OnlineProvider(api_key="sk-xxx", model="gpt-4")
    result = provider.generate("你好")
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model_name = model
        self._device = "cloud"
        self._client = None

        if self._api_key:
            try:
                # openai SDK >= 1.0 的客户端（DeepSeek 等 OpenAI 兼容服务同样适用）
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                )
            except ImportError:
                self._client = None

    @property
    def model_name(self) -> str:
        """返回当前使用的模型名称"""
        return self._model_name
    
    @property
    def device(self) -> str:
        """返回当前设备信息"""
        return self._device

    @property
    def api_key(self) -> Optional[str]:
        """获取 API Key（脱敏）"""
        if self._api_key and len(self._api_key) > 8:
            return f"{self._api_key[:4]}...{self._api_key[-4:]}"
        return self._api_key

    def generate(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None, 
        **kwargs
    ) -> str:
        """
        调用 OpenAI API 生成文本
        
        Args:
            prompt: 输入提示词
            images: 图片列表（当前不支持）
            **kwargs: 额外参数
            
        Returns:
            生成的文本字符串
        """
        if images:
            raise ValueError("当前仅支持文本输入")

        if self._client is None:
            logger.error("OpenAI SDK 未安装或 API Key 未设置，无法生成文本")
            return "[openai-client-unavailable] " + prompt

        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API 调用失败：{e}")
            raise RuntimeError(f"OpenAI API 调用失败：{e}") from e

    def generate_with_thoughts(
        self, 
        prompt: str, 
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成（OpenAI 简化实现）
        
        注意：当前版本为简化实现，完整版应使用流式 API
        """
        try:
            result = self.generate(prompt, **kwargs)
            yield {"type": "complete", "full_text": result}
        except Exception as e:
            yield {"type": "error", "content": str(e)}

    def stop_generation(self):
        """
        停止生成（在线模型通常不支持停止）
        
        对于 OpenAI，可以通过取消请求实现
        """
        logger.info("Online provider: stop_generation called (not fully supported)")

    def generate_with_image(
        self, 
        prompt: str, 
        image_path: str, 
        **kwargs
    ) -> str:
        """
        生成带图片的响应（需要 GPT-4V 支持）
        
        TODO: 实现 GPT-4 Vision 集成
        """
        raise NotImplementedError("在线模型图片生成功能开发中")


# 向后兼容：保留 OpenAIAgent 作为 OnlineProvider 的别名
OpenAIAgent = OnlineProvider
