"""
Base Provider - 模型提供者抽象基类

职责：
- 定义模型提供者的统一接口
- 规范生成、流式生成、停止等核心方法
- 为 LocalProvider 和 OnlineProvider 提供契约

设计原则：
- 接口隔离：只定义必要的方法
- 依赖倒置：高层模块依赖此抽象，不依赖具体实现
- 开放封闭：对扩展开放（新的 Provider），对修改封闭
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Generator, Any


class BaseProvider(ABC):
    """
    模型提供者基类
    
    所有模型提供者（Local、OpenAI、Anthropic 等）都必须实现此接口
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
