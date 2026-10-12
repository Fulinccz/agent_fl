"""通用输出格式化工具

规则式（零 LLM 开销）排版归一化，供润色等链路在输出前调用；
可复用于任何需要整洁纯文本排版的 LLM 输出场景。
"""
from __future__ import annotations

import re


class OutputFormatter:
    """通用 LLM 输出格式化器（规则式，零 LLM 开销）

    用法：
        from agents.skills.output_formatter import OutputFormatter
        OutputFormatter().format(text)                # 简历排版（默认）
        OutputFormatter().format(text, doc_type="plain")  # 仅基础归一化
    """

    # 独立成行的常见章节关键词 → 视为节标题（仅匹配"标题："形态，不带正文）
    _SECTION_RE = re.compile(
        r'^(个人信息|基本资料|基本信息|求职意向|教育背景|教育经历|工作经历|工作经验|'
        r'项目经历|项目经验|实习经历|校园经历|技能清单|专业技能|技能特长|核心技能|'
        r'资格证书|证书荣誉|荣誉奖项|获奖情况|自我评价|自我介绍|兴趣爱好|其他)[:：]?\s*$'
    )

    # 时间开头的行（公司/学校条目），如 2020.07-至今 / 2017.07-2020.06 / 2018年3月
    _DATE_LINE_RE = re.compile(r'^\d{4}\s*[.\-/年]')

    # 各类 bullet 符号 → 统一为 "- "
    _BULLET_RE = re.compile(r'^\s*[•·◆●◦*–—‐‣⁃]\s+')

    # 行首残留 markdown 标记
    _MD_HEADER_RE = re.compile(r'^#{1,6}\s*')
    _BOLD_RE = re.compile(r'\*\*')

    def format(self, text: str, doc_type: str = "resume") -> str:
        """格式化入口；doc_type="resume" 走简历排版，其余仅做基础归一化"""
        if not text or not text.strip():
            return ""
        lines = self._normalize_lines(text)
        if doc_type == "resume":
            lines = self._layout_resume(lines)
        return self._collapse_blanks(lines)

    # ---------- 基础归一化 ----------

    def _normalize_lines(self, text: str) -> list:
        """去 markdown 残留、统一 bullet 符号、去行尾空白"""
        lines = []
        for raw in text.split('\n'):
            line = self._BOLD_RE.sub('', raw)
            line = self._MD_HEADER_RE.sub('', line).rstrip()
            line = self._BULLET_RE.sub('- ', line)
            lines.append(line)
        return lines

    @staticmethod
    def _collapse_blanks(lines: list) -> str:
        """压缩 3+ 连续换行与首尾空行"""
        out = []
        blanks = 0
        for line in lines:
            if line.strip() == '':
                blanks += 1
                if blanks <= 1:
                    out.append('')
            else:
                blanks = 0
                out.append(line)
        return '\n'.join(out).strip()

    # ---------- 简历排版 ----------

    def _layout_resume(self, lines: list) -> list:
        """节标题前留空行；时间条目与上一条目之间留空行"""
        out: list = []
        prev_is_header = False
        for line in lines:
            stripped = line.strip()

            if not stripped:
                out.append('')
                prev_is_header = False
                continue

            m = self._SECTION_RE.match(stripped)
            if m:
                # 节标题：统一「标题：」形态，前面保证空行
                if out and out[-1].strip():
                    out.append('')
                out.append(m.group(1) + '：')
                prev_is_header = True
                continue

            if self._DATE_LINE_RE.match(stripped) and not prev_is_header:
                # 新的时间条目：与上一条目之间留空行
                if out and out[-1].strip():
                    out.append('')
                out.append(line)
                prev_is_header = False
                continue

            out.append(line)
            prev_is_header = False
        return out
