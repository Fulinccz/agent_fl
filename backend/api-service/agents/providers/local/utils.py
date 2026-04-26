"""
Local Provider 工具函数
包含内容过滤、文本处理等辅助功能
"""

from __future__ import annotations
import re
from typing import List


def filter_think_content(content: str) -> str:
    """过滤思考内容中的 Prompt 指令重复"""
    patterns = [
        r'按以下\d*个部分输出',
        r'每部分用【标题】开头',
        r'Each part must start with',
        r'Output according to',
        r'indicat.*prompt',
        r'marker.*be formatted',
        r'looking closely at the instruction',
        r'The instruction says',
        r'【简历评分】.*?只给分数',
        r'【优化建议】.*?简短列出',
        r'【优化结果】.*?重写一版',
        r'先用.*结束',
        r'之后按以下',
    ]
    
    lines = content.split('\n')
    filtered = []
    for line in lines:
        should_skip = False
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                should_skip = True
                break
        if not should_skip and line.strip():
            filtered.append(line)
    
    return '\n'.join(filtered)


def check_stop_patterns(text: str, stop_patterns: List[str]) -> bool:
    """检查文本是否包含停止模式"""
    for pattern in stop_patterns:
        if pattern in text:
            return True
    return False
