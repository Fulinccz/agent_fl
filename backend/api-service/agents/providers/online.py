"""
Online Provider - OpenAI API 提供者

职责：
- 封装 OpenAI API 调用
- 实现统一的提供者接口
- 支持在线模型推理

设计原则：
- 继承 BaseProvider，与 LocalProvider 统一接口
- 优雅降级：API Key 未设置时返回提示信息
"""

from __future__ import annotations

from logger import get_logger
import os
from typing import Optional, List, Dict, Generator, Any

from ..core.base_provider import BaseProvider

logger = get_logger(__name__)


class OnlineProvider(BaseProvider):
    """
    OpenAI 在线模型提供者
    
    特性：
    - ✅ 支持 OpenAI GPT 系列
    - ✅ 统一的提供者接口
    - ⚠️ 需要 API Key
    
    使用方式：
        provider = OnlineProvider(api_key="sk-xxx", model="gpt-4")
        result = provider.generate("你好")
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        try:
            import openai
        except ImportError:
            openai = None

        self._openai = openai
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model_name = model
        self._device = "cloud"

        if self._openai and self._api_key:
            self._openai.api_key = self._api_key

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

        if self._openai is None:
            logger.error("OpenAI SDK 未安装，无法生成文本")
            return "[openai-sdk-not-installed] " + prompt

        if not self._api_key:
            logger.error("OpenAI API Key 未设置")
            return "[openai-api-key-not-set] " + prompt

        resp = self._openai.ChatCompletion.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content

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
