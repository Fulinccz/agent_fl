"""
Online Agent - 向后兼容的导出

实际实现已迁移到：providers/online.py

此文件保留用于向后兼容，所有导入都会自动重定向到新位置。
"""

from __future__ import annotations

# 从新的位置导入
from .providers.online import OnlineProvider, OpenAIAgent

# 导出所有公共接口
__all__ = ['OnlineProvider', 'OpenAIAgent']
