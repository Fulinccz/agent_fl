"""
Base Provider
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Generator, Any


class BaseProvider(ABC):
    """
    模型提供者基类
    """
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None, 
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            images: 图片列表（可选）
            **kwargs: 额外参数（temperature, max_tokens 等）
            
        Returns:
            生成的文本字符串
        """
        pass
    
    @abstractmethod
    def generate_with_thoughts(
        self, 
        prompt: str, 
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成文本，支持思考过程
        
        Yields:
            包含 type 和 content 的字典
        """
        pass
    
    @abstractmethod
    def stop_generation(self):
        """停止当前生成任务"""
        pass
    
    @abstractmethod
    def generate_with_image(
        self, 
        prompt: str, 
        image_path: str, 
        **kwargs
    ) -> str:
        """生成带图片的响应"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前使用的模型名称"""
        pass
    
    @property
    @abstractmethod  
    def device(self) -> str:
        """返回当前设备信息"""
        pass
