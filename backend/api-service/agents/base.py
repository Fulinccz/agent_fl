"""
Agent Base Classes - 向后兼容的基类导出

此文件保留用于向后兼容，实际实现已迁移到：
- core/base_provider.py: BaseProvider (模型提供者基类)
- core/base_agent.py: BaseAgent (多技能 Agent 基类)
- core/base_tool.py: BaseTool, ToolResult (工具基类)
"""

from __future__ import annotations

# 导出核心基类（统一接口）
from .core.base_provider import BaseProvider
from .core.base_agent import BaseAgent as CoreBaseAgent
from .core.base_tool import BaseTool, ToolResult, ToolRegistry

# 向后兼容：BaseAgent 指向 BaseProvider（因为原来的 LocalAgent 实际上是 Provider）
BaseAgent = BaseProvider

__all__ = [
    'BaseProvider',
    'BaseAgent',  # 别名，指向 BaseProvider
    'CoreBaseAgent',  # 真正的 Agent 基类（用于 LangChain）
    'BaseTool',
    'ToolResult',
    'ToolRegistry',
]
