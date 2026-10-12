"""优化简历文档生成服务

双路径策略：
- 用户上传过 docx：复制原文件，按内容对齐替换改写（保留图标/图片/表格/页面设置等全部格式）
- 无上传或非 docx：走 ResumeDocxSkill 模板重新排版生成

导出文件写入系统临时目录，由路由在响应发送后删除。
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
            return DocxExportResult(path=out_path, filename=filename, mode="rewrite")
        except Exception as e:
            logger.warning(f"[resume_document] 原文件改写失败，回退模板生成: {e}")

    from agents.skills.doc_generation import ResumeDocxSkill
    ResumeDocxSkill().generate(text, out_path)
    logger.info(f"[resume_document] 模板生成完成: {filename}")
    return DocxExportResult(path=out_path, filename=filename, mode="template")


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

    polished_lines = [l.strip() for l in polished_text.split("\n") if l.strip()]
    norm_polished = {i: _normalize(l) for i, l in enumerate(polished_lines)}

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
