"""
Base Agent - Agent 抽象基类（LangChain 预留）

职责：
- 定义多技能 Agent 的统一接口
- 支持工具调用和工作流编排
- 为未来的 LangChain 集成预留接口

设计原则：
- 组合优于继承：Agent 通过组合使用 Tools 和 Provider
- 策略模式：可切换不同的推理策略
- 观察者模式：支持回调机制（用于流式输出、日志等）

未来集成：
- LangChain Agent
- ReAct Agent
- Plan-and-Execute Agent
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Generator
from .base_tool import BaseTool, ToolRegistry


class BaseAgent(ABC):
    """
    Agent 基类
    
    多技能 Agent 的抽象定义，支持：
    - 工具注册和管理
    - 任务规划和执行
    - 上下文管理
    """
    
    def __init__(
        self, 
        provider=None,
        tools: Optional[List[BaseTool]] = None,
        name: str = "base_agent",
        description: str = ""
    ):
        """
        初始化 Agent
        
        Args:
            provider: 模型提供者
            tools: 初始工具列表
            name: Agent 名称
            description: Agent 描述
        """
        self._provider = provider
        self._tools: List[BaseTool] = tools or []
        self.name = name
        self.description = description
        self._context: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
    
    @property
    def provider(self):
        """获取模型提供者"""
        return self._provider
    
    def add_tool(self, tool: BaseTool):
        """添加工具"""
        self._tools.append(tool)
        return self
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """按名称获取工具"""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
    
    @abstractmethod
    def run(
        self, 
        task: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务描述
            **kwargs: 额外参数
            
        Returns:
            任务执行结果
        """
        pass
    
    @abstractmethod
    async def arun(
        self, 
        task: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """异步执行任务"""
        pass
    
    def stream(
        self, 
        task: str, 
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式执行任务
        
        Yields:
            中间结果和最终结果
        """
        raise NotImplementedError("Subclass must implement stream()")
    
    def set_context(self, key: str, value: Any):
        """设置上下文"""
        self._context[key] = value
    
    def get_context(self, key: str, default=None) -> Any:
        """获取上下文"""
        return self._context.get(key, default)
    
    def add_to_history(self, entry: Dict[str, Any]):
        """添加到历史记录"""
        self._history.append(entry)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self._history.copy()
    
    def clear_history(self):
        """清除历史记录"""
        self._history.clear()
    
    def list_tools(self) -> List[str]:
        """列出所有可用工具名称"""
        return [tool.name for tool in self._tools]
