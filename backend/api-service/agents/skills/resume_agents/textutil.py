"""简历长文本工具

- split_text: 按段落边界把超长简历切成不超过指定字数的块，
  供分段精简/分段润色使用，避免单次 prompt 截断导致简历尾部内容丢失。
- strip_llm_artifacts: 移除 LLM 输出的通用杂质（think 标签、代码围栏）。
"""
from __future__ import annotations

import re
from typing import List

# 句末标点（超长单段落二次切分用）
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;!?])")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n?")


def strip_llm_artifacts(text: str) -> str:
    """移除 LLM 输出的通用杂质：<think> 标签与 markdown 代码围栏。

    condense/polish 等多个 Agent 的 _clean_result 公共内核，
    各自特有的清理逻辑（前缀行/角色标记等）由调用方继续处理。
    """
    text = _THINK_RE.sub("", text or "")
    text = _FENCE_RE.sub("", text)
    return text.replace("```", "")


def split_text(text: str, limit: int) -> List[str]:
    """按段落（换行）边界切分，每块不超过 limit 字。

    - 正常情况：贪心累积段落
    - 单个段落超长：按句末标点再切；单句仍超长则硬切
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    segments: List[str] = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            segments.append(buffer.strip())
        buffer = ""

    for para in text.split("\n"):
        if len(para) <= limit:
            if buffer and len(buffer) + 1 + len(para) > limit:
                flush()
            buffer = f"{buffer}\n{para}" if buffer else para
        else:
            # 超长段落：先冲刷已有缓冲，再按句子切
            flush()
            for piece in _split_long_paragraph(para, limit):
                segments.append(piece)
    flush()
    return segments


def _split_long_paragraph(para: str, limit: int) -> List[str]:
    """超长段落：优先按句末标点切，单句仍超则硬切"""
    pieces: List[str] = []
    buf = ""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(para) if s]
    if len(sentences) == 1:
        # 没有句末标点：硬切
        return [para[i:i + limit] for i in range(0, len(para), limit)]

    for sent in sentences:
        if len(sent) > limit:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(sent[i:i + limit] for i in range(0, len(sent), limit))
        elif buf and len(buf) + len(sent) > limit:
            pieces.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        pieces.append(buf)
    return pieces
