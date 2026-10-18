"""LLM 输出 JSON 提取工具

从 LLM 自由文本中提取第一个平衡的 JSON 对象。

背景：直接用 re.search(r"\{.*\}", text, re.DOTALL) 贪心匹配在以下场景会失败：
- 文本中含多个 JSON 对象（截取范围错误）
- 字符串值内含花括号（如建议文本 "请补充 {项目名}"）
- 多层嵌套（如评分 analysis → 条目对象 → suggestion 字段）
因此改用字符串感知的平衡括号扫描，与 score_agent 历史实现一致。
"""
from __future__ import annotations

from typing import Optional


def extract_json_object(text: str) -> Optional[str]:
    """提取文本中第一个平衡的 JSON 对象（字符串感知，支持任意嵌套深度）

    返回 JSON 字符串片段（含首尾花括号），未找到返回 None。
    仅做括号平衡扫描，不校验 JSON 合法性；调用方自行 json.loads。
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
