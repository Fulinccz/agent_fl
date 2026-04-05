"""
Base Tool - 工具抽象基类

职责：
- 定义工具的统一接口
- 实现工具注册和发现机制
- 为所有业务工具提供标准化的调用方式

设计原则（参考 LangChain Tools）：
- 每个工具只做一件事（单一职责）
- 工具可插拔、可组合
- 统一的输入输出格式
- 支持自动文档生成

使用示例：
    class ResumeParserTool(BaseTool):
        name = "resume_parser"
        description = "解析简历，提取结构化信息"
        
        def execute(self, resume_content: str) -> Dict:
            # 实现逻辑
            return parsed_data
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """工具执行结果的标准格式"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "success": self.success,
            "data": self.data,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseTool(ABC):
    """
    工具基类
    
    所有业务工具都必须继承此类并实现 execute 方法
    """
    
    name: str = ""  # 工具名称（唯一标识）
    description: str = ""  # 工具描述（用于 Agent 选择）
    
    def __init__(self, provider=None):
        """
        初始化工具
        
        Args:
            provider: 模型提供者实例（用于调用 LLM）
        """
        self._provider = provider
        self._validate()
    
    def _validate(self):
        """验证工具定义是否完整"""
        if not self.name:
            raise ValueError(f"Tool {self.__class__.__name__} must have a 'name' attribute")
        if not self.description:
            raise ValueError(f"Tool {self.__class__.__name__} must have a 'description' attribute")
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具逻辑
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult 标准化结果
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        获取工具的 JSON Schema（用于 LangChain 集成）
        
        Returns:
            包含工具元信息的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "class": self.__class__.__name__,
        }
    
    def __repr__(self):
        return f"<Tool: {self.name}>"


class ToolRegistry:
    """
    工具注册表（单例模式）
    
    管理所有可用工具的注册、发现和调用
    """
    
    _instance = None
    _tools: Dict[str, Type[BaseTool]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, tool_class: Type[BaseTool]):
        """
        注册工具类（装饰器用法）
        
        用法：
            @ToolRegistry.register
            class MyTool(BaseTool):
                ...
        """
        instance = tool_class()
        cls._tools[instance.name] = tool_class
        return tool_class
    
    @classmethod
    def get_tool(cls, name: str, provider=None) -> Optional[BaseTool]:
        """
        获取工具实例
        
        Args:
            name: 工具名称
            provider: 模型提供者
            
        Returns:
            工具实例，如果不存在返回 None
        """
        tool_class = cls._tools.get(name)
        if tool_class:
            return tool_class(provider=provider)
        return None
    
    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]:
        """
        列出所有已注册的工具
        
        Returns:
            工具信息列表
        """
        return [
            tool_class().get_schema() 
            for tool_class in cls._tools.values()
        ]
    
    @classmethod
    def get_tools_by_category(cls, category: str) -> List[BaseTool]:
        """
        按类别获取工具（基于命名约定）
        
        Args:
            category: 类别名称（如 'resume', 'text'）
            
        Returns:
            该类别下的工具列表
        """
        tools = []
        for name, tool_class in cls._tools.items():
            if category.lower() in name.lower():
                tools.append(tool_class())
        return tools
