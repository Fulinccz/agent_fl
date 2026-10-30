"""优化简历文档生成服务

双路径策略：
- 用户上传过 docx：复制原文件，按内容对齐替换改写（保留图标/图片/表格/页面设置等全部格式）
- 无上传或非 docx：走 ResumeDocxSkill 模板重新排版生成

导出文件写入导出目录，由路由登记 token 后通过 URL 复用（含只读预览），
源文件（用户上传的临时副本）在生成完成后即刻删除。
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher

from logger import get_logger

logger = get_logger(__name__)

# 对齐替换的最小相似度阈值（归一化编辑相似度）
_ALIGN_THRESHOLD = 0.35


@dataclass
class DocxExportResult:
    path: str
    filename: str
    mode: str  # "rewrite"（原文件改写） / "template"（模板生成）
    size: int = 0


def _sanitize_base(name: str | None) -> str | None:
    """原文件名 → 安全的文档名主体（去扩展名与路径分隔符）"""
    if not name:
        return None
    base = os.path.basename(name)
    base = os.path.splitext(base)[0].strip()
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", base)
    return base[:60] or None


def sanitize_filename(name: str | None, default: str = "file") -> str:
    """上传文件名消毒：去目录成分与 Windows 非法字符，保留扩展名。

    供上传/导出落盘前调用，防止路径穿越与特殊字符导致写盘异常。
    """
    if not name:
        return default
    base = os.path.basename(name).replace("\x00", "").strip()
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", base)
    base = base.strip(". ") or default
    return base[:120]


def generate_resume_docx(text: str, source_path: str | None = None,
                         source_name: str | None = None) -> DocxExportResult:
    """生成优化简历 docx。

    - source_path：用户上传的原始文件（仅 .docx 支持改写路径）
    - source_name：原始文件名（用于输出命名）
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("优化结果为空，无法生成文档")

    base = _sanitize_base(source_name)
    filename = f"{base}_优化版.docx" if base else "简历_优化版.docx"

    export_dir = os.path.join(tempfile.gettempdir(), "fulin_exports")
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, filename)

    # 同名冲突：追加序号
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(filename)
        n = 1
        while os.path.exists(os.path.join(export_dir, f"{stem}_{n}{ext}")):
            n += 1
        out_path = os.path.join(export_dir, f"{stem}_{n}{ext}")
        filename = os.path.basename(out_path)

    is_docx = bool(source_path) and source_path.lower().endswith(".docx") \
        and os.path.exists(source_path)

    if is_docx:
        try:
            _rewrite_docx_aligned(source_path, text, out_path)
            logger.info(f"[resume_document] 原文件改写完成: {filename}")
            return DocxExportResult(
                path=out_path, filename=filename, mode="rewrite",
                size=os.path.getsize(out_path),
            )
        except Exception as e:
            logger.warning(f"[resume_document] 原文件改写失败，回退模板生成: {e}")

    from agents.skills.doc_generation import ResumeDocxSkill
    ResumeDocxSkill().generate(text, out_path)
    logger.info(f"[resume_document] 模板生成完成: {filename}")
    return DocxExportResult(
        path=out_path, filename=filename, mode="template",
        size=os.path.getsize(out_path),
    )


# ---------- 对齐替换改写 ----------

def _normalize(s: str) -> str:
    """对齐用归一化：去空白与标点，转小写"""
    return re.sub(r"[\s，。；：、！？（）()【】\[\]\"'“”‘’·—\-–.·:;,!?()<>|/\\]+", "",
                  (s or "").lower())


def _set_paragraph_text(para, new_text: str):
    """替换段落文本，保留段落样式与首个 run 的字体格式"""
    if para.runs:
        para.runs[0].text = new_text
        for r in para.runs[1:]:
            r.text = ""
    else:
        para.add_run(new_text)


def _rewrite_docx_aligned(source_path: str, polished_text: str, out_path: str):
    """复制原文档并按内容对齐替换文本段落。

    - 能对齐的段落：替换文本（保留段落样式、字体、图标 run 结构）
    - 对不齐的原段落：保留不动（图标、证书行等不丢）
    - 对不齐的润色行：追加到文末（避免内容缺失）
    """
    from docx import Document

    shutil.copyfile(source_path, out_path)
    doc = Document(out_path)

    polished_lines = [line.strip() for line in polished_text.split("\n") if line.strip()]
    norm_polished = {i: _normalize(line) for i, line in enumerate(polished_lines)}

    body_paras = [p for p in doc.paragraphs if p.text.strip()]
    norm_body = {id(p): _normalize(p.text) for p in body_paras}

    # 全量打分 → 贪心一对一配对（相似度降序，避免多段抢同一行）
    scores = []
    for pi, p in enumerate(body_paras):
        np_ = norm_body[id(p)]
        if not np_:
            continue
        for li, nl in norm_polished.items():
            if not nl:
                continue
            r = SequenceMatcher(None, np_, nl).ratio()
            if r >= _ALIGN_THRESHOLD:
                scores.append((r, pi, li))
    scores.sort(reverse=True)

    used_paras: set[int] = set()
    used_lines: set[int] = set()
    matched: list[tuple[int, int]] = []  # (para_index, line_index)
    for r, pi, li in scores:
        if pi in used_paras or li in used_lines:
            continue
        used_paras.add(pi)
        used_lines.add(li)
        matched.append((pi, li))

    for pi, li in matched:
        _set_paragraph_text(body_paras[pi], polished_lines[li])

    # 未对齐的润色行：插入到对齐锚点之后，无锚点则追加文末
    leftover = [polished_lines[li] for li in sorted(set(norm_polished) - used_lines)]
    if leftover:
        anchor = body_paras[max(pi for pi, _ in matched)] if matched else None
        _insert_after(anchor, leftover, doc)

    doc.save(out_path)


def _insert_after(anchor_para, lines: list[str], doc):
    """在锚点段落后按顺序插入新段落。

    优先继承锚点段落自身样式（插入位置通常是正文区）；
    无锚点时退化为文档首段样式或默认段落。
    """
    from copy import deepcopy
    from docx.text.paragraph import Paragraph

    W_NS = ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r"
    ref = anchor_para
    for line in lines:
        if ref is not None:
            # 继承锚点（正文）样式，避免误用文档首段标题样式
            new_p = deepcopy(ref._p)
            for r in list(new_p.findall(W_NS)):
                new_p.remove(r)
            ref._p.addnext(new_p)
            para = Paragraph(new_p, ref._parent)
            para.add_run(line)
            ref = para
        else:
            para = doc.add_paragraph(line)
            ref = para


# ---------- 只读预览 ----------

_PREVIEW_CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: #f3f4f6; color: #1f2430;
  font-family: "Microsoft YaHei", "PingFang SC", -apple-system, sans-serif; }
.resume-paper { max-width: 760px; margin: 0 auto; background: #fff; padding: 34px 42px;
  border-radius: 6px; box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08); }
h2 { font-size: 15px; margin: 18px 0 8px; padding-bottom: 4px;
  border-bottom: 1px solid #e5e7eb; }
p { margin: 5px 0; font-size: 13px; line-height: 1.75;
  white-space: pre-wrap; word-break: break-word; }
p.bullet { padding-left: 16px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12.5px; }
td { border: 1px solid #e5e7eb; padding: 5px 8px; vertical-align: top; }
.empty { color: #9ca3af; text-align: center; padding: 40px 0; }
"""


def render_docx_preview_html(path: str) -> str:
    """把 docx 渲染为只读 HTML 预览（段落 / 标题 / 表格 / 图标行文本）

    只做服务端渲染，前端以 sandbox iframe 展示，不提供任何编辑入口。
    """
    from html import escape

    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(path)
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            if _is_heading(para):
                blocks.append(f"<h2>{escape(text)}</h2>")
            else:
                cls = ' class="bullet"' if text.startswith(("•", "-", "·")) else ""
                blocks.append(f"<p{cls}>{escape(text)}</p>")
        elif tag == "tbl":
            blocks.extend(_render_table_html(Table(child, doc), escape))

    inner = "\n".join(blocks) or '<p class="empty">（文档内容为空）</p>'
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>简历预览</title><style>{_PREVIEW_CSS}</style></head>"
        f'<body><div class="resume-paper">{inner}</div></body></html>'
    )


def _is_heading(para) -> bool:
    """按段落样式判断是否标题（样式名缺失时按正文处理）"""
    try:
        name = para.style.name or ""
    except Exception:
        return False
    return "Heading" in name or "标题" in name


def _render_table_html(table, escape) -> list[str]:
    rows: list[str] = []
    for row in table.rows:
        cells = "".join(f"<td>{escape(c.text.strip())}</td>" for c in row.cells)
        rows.append(f"<tr>{cells}</tr>")
    if not rows:
        return []
    return [f'<table>{"".join(rows)}</table>']
