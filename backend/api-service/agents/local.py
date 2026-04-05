"""
Local Agent - 向后兼容的导出

实际实现已迁移到：providers/local.py

此文件保留用于向后兼容，所有导入都会自动重定向到新位置。
"""

from __future__ import annotations

# 从新的位置导入
from .providers.local import LocalProvider, LocalAgent

# 导出所有公共接口
__all__ = ['LocalProvider', 'LocalAgent']
