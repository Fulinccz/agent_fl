"""简历文档生成 Skill

将纯文本简历渲染为排版规范的 .docx（无上传文件时的模板路径）。
独立成 Skill 便于维护与复用：节标题/条目/正文样式集中在此调整。
"""
from __future__ import annotations

import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from agents.skills.output_formatter import OutputFormatter


class ResumeDocxSkill:
    """纯文本简历 → 排版规范 docx（零 LLM，规则渲染）"""

    name = "resume_docx"
    description = "将纯文本简历渲染为排版规范的 Word 文档（模板路径）"

    _SECTION_RE = OutputFormatter._SECTION_RE
    _BULLET_RE = re.compile(r"^[-•·◆●◦*–—‐‣⁃]\s*")
    _DATE_LINE_RE = re.compile(r"^\d{4}\s*[.\-/年]")

    # 配色与字号（集中定义便于维护）
    NAME_SIZE = Pt(16)
    SECTION_SIZE = Pt(12)
    BODY_SIZE = Pt(10.5)
    ACCENT = RGBColor(0x1F, 0x4E, 0x79)
    TEXT = RGBColor(0x26, 0x26, 0x26)

    def generate(self, text: str, output_path: str) -> str:
        doc = Document()
        self._setup_page(doc)

        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError("简历文本为空，无法生成文档")

        self._add_name(doc, lines[0])
        for line in lines[1:]:
            self._add_line(doc, line)

        doc.save(output_path)
        return output_path

    # ---------- 页面与样式 ----------

    @staticmethod
    def _setup_page(doc: Document):
        for section in doc.sections:
            section.top_margin = Cm(2.0)
            section.bottom_margin = Cm(2.0)
            section.left_margin = Cm(2.2)
            section.right_margin = Cm(2.2)

    def _set_font(self, run, size: Pt, bold: bool = False, color: RGBColor | None = None):
        run.font.name = "Calibri"
        run.font.size = size
        run.font.bold = bold
        run.font.color.rgb = color or self.TEXT
        # 中文字体需单独设置 eastAsia
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    def _add_name(self, doc: Document, name_line: str):
        """首行作为姓名/标题：居中加大加粗"""
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(name_line)
        self._set_font(run, self.NAME_SIZE, bold=True, color=self.ACCENT)
        p.paragraph_format.space_after = Pt(10)

    def _add_line(self, doc: Document, line: str):
        m = self._SECTION_RE.match(line)
        if m:
            p = doc.add_paragraph()
            run = p.add_run(m.group(1))
            self._set_font(run, self.SECTION_SIZE, bold=True, color=self.ACCENT)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            # 标题下划线（浅色边框）
            pPr = p._p.get_or_add_pPr()
            borders = pPr.makeelement(qn("w:pBdr"), {})
            bottom = pPr.makeelement(qn("w:bottom"), {
                qn("w:val"): "single", qn("w:sz"): "6",
                qn("w:space"): "2", qn("w:color"): "1F4E79",
            })
            borders.append(bottom)
            pPr.append(borders)
            return

        if self._BULLET_RE.match(line):
            p = doc.add_paragraph()
            run = p.add_run("• " + self._BULLET_RE.sub("", line))
            self._set_font(run, self.BODY_SIZE)
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.25
            return

        p = doc.add_paragraph()
        run = p.add_run(line)
        # 时间条目（公司/学校行）加粗突出
        bold = bool(self._DATE_LINE_RE.match(line))
        self._set_font(run, self.BODY_SIZE, bold=bold)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.25
