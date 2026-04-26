"""
Online Agent
"""

from __future__ import annotations

# 从新的位置导入
from .providers.online import OnlineProvider, OpenAIAgent

# 导出所有公共接口
__all__ = ['OnlineProvider', 'OpenAIAgent']
