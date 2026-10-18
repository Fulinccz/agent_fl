"""
1) ResumeParser：支持 .docx/.doc/.pdf 简历解析，提取技能栏和项目描述
2) DocumentProcessor：知识库文档切片器，按 markdown 标题切 chunk 供向量入库
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResumeParser:
    def __init__(self):
        """初始化解析器"""
        # 技能栏关键词
        self.skill_keywords = [
            r'专业技能', r'技能清单', r'技术栈', r'技能栏',
            r'专业特长', r'技术能力', r'掌握技能', r'熟悉技术',
            r'Skills', r'Technical Skills', r'Professional Skills',
        ]
        
        # 项目描述关键词
        self.project_keywords = [
            r'项目经历', r'项目经验', r'项目描述', r'项目案例',
            r'项目介绍', r'代表项目', r'主要项目', r'项目成果',
            r'Project', r'Projects', r'Project Experience',
        ]
        
        # 通用段落标记
        self.section_markers = [
            r'^[-•●○■◆▪▸]\s*',  # 列表符号
            r'^\d+[\.、]\s*',     # 数字编号
        ]

    def parse_file(self, file_path: str) -> Dict[str, Optional[str]]:
        """
        解析简历文件
        """
        file_path = Path(file_path)
        logger.info(f"开始解析简历文件：{file_path}")
        
        if not file_path.exists():
            logger.error(f"文件不存在：{file_path}")
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        file_extension = file_path.suffix.lower()
        
        # 检测是否为假 .docx（实际是 .doc）
        if file_extension == '.docx':
            import zipfile
            if not zipfile.is_zipfile(str(file_path)):
                logger.warning("文件不是真正的 .docx 格式，尝试作为 .doc 处理")
                file_extension = '.doc'
        
        logger.info(f"文件格式：{file_extension}")
        
        if file_extension == '.docx':
            content = self._parse_docx(file_path)
        elif file_extension == '.doc':
            content = self._parse_doc(file_path)
        elif file_extension == '.pdf':
            content = self._parse_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{file_extension}")
        
        # 提取技能栏和项目描述
        skills = self._extract_skills(content)
        projects = self._extract_projects(content)
        
        logger.info(f"简历解析完成：技能栏{len(skills) if skills else 0}字，项目描述{len(projects) if projects else 0}字")
        
        return {
            "skills": skills,
            "projects": projects,
            "full_content": content,
        }

    def _parse_docx(self, file_path: Path) -> str:
        """解析 .docx 文件"""
        try:
            from docx import Document
            doc = Document(file_path)
            
            paragraphs = []
            
            # 提取段落
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            
            # 检查文本框（Shape）
            if hasattr(doc, 'inline_shapes') and doc.inline_shapes:
                for shape in doc.inline_shapes:
                    try:
                        if hasattr(shape, 'text'):
                            text = shape.text.strip()
                            if text:
                                paragraphs.append(text)
                    except Exception:
                        pass
            
            result = '\n'.join(paragraphs)
            
            # 如果 python-docx 没读到内容，尝试用 mammoth
            if len(result) == 0:
                try:
                    import mammoth
                    with open(file_path, "rb") as f:
                        result_mammoth = mammoth.extract_raw_text(f)
                        if result_mammoth.value:
                            logger.info(f"mammoth 提取成功：{len(result_mammoth.value)} 字符")
                            return result_mammoth.value
                except Exception as e:
                    logger.warning(f"mammoth 解析失败：{e}")
            
            return result
        except Exception as e:
            logger.error(f"解析 docx 失败：{e}")
            raise ValueError(f"无法解析 .docx 文件：{e}")

    def _parse_doc(self, file_path: Path) -> str:
        """解析 .doc 文件（通过 Word COM 接口）"""
        try:
            import win32com.client
            
            logger.info("使用 Word COM 接口解析 .doc 文件")
            
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            
            abs_path = str(file_path.absolute())
            logger.info(f"打开文件：{abs_path}")
            
            doc = word.Documents.Open(abs_path)
            
            paragraphs = []
            logger.info(f"段落数：{doc.Paragraphs.Count}")
            
            for i, para in enumerate(doc.Paragraphs):
                text = para.Range.Text.strip()
                if text:
                    logger.info(f"段落 {i+1}: {text[:100]}")
                    paragraphs.append(text)
            
            doc.Close(False)
            word.Quit()
            
            result = '\n'.join(paragraphs)
            logger.info(f"doc 解析完成，共 {len(paragraphs)} 个段落，总长度 {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"解析 doc 失败：{e}")
            raise ValueError(f"无法解析 .doc 文件：{e}")

    def _parse_pdf(self, file_path: Path) -> str:
        """解析 .pdf 文件"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            
            return '\n'.join(pages)
        except Exception as e:
            logger.error(f"解析 pdf 失败：{e}")
            raise ValueError(f"无法解析 .pdf 文件：{e}")

    def _extract_skills(self, content: str) -> Optional[str]:
        """提取技能栏内容"""
        sections = self._split_sections(content)
        
        # 查找技能相关章节
        for title, section_content in sections.items():
            if any(re.search(keyword, title, re.IGNORECASE) for keyword in self.skill_keywords):
                return self._clean_section(section_content)
        
        # 未找到明确章节，返回空
        return None

    def _extract_projects(self, content: str) -> Optional[str]:
        """提取项目描述内容"""
        sections = self._split_sections(content)
        
        # 查找项目相关章节
        for title, section_content in sections.items():
            if any(re.search(keyword, title, re.IGNORECASE) for keyword in self.project_keywords):
                return self._clean_section(section_content)
        
        logger.debug("未找到明确的项目章节，尝试从内容中查找关键词")
        
        # 如果没有找到明确的项目章节，尝试从内容中查找包含项目关键词的部分
        lines = content.split('\n')
        project_lines = []
        in_project_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是项目相关行
            if any(re.search(keyword, line, re.IGNORECASE) for keyword in self.project_keywords):
                logger.debug(f"在第 {i} 行找到项目关键词：'{line[:50]}'")
                in_project_section = True
                continue
            
            # 如果在项目部分，收集内容
            if in_project_section:
                # 检查是否遇到新的章节标题
                if self._is_section_title(line):
                    logger.debug(f"遇到新章节标题，停止收集项目：'{line[:50]}'")
                    break
                project_lines.append(line)
        
        if project_lines:
            logger.info(f"通过关键词找到项目内容，共 {len(project_lines)} 行")
            return '\n'.join(project_lines)
        
        logger.debug("未找到项目关键词，返回空")
        return None

    def _split_sections(self, content: str) -> Dict[str, str]:
        """
        将内容按章节分割
        返回：{章节标题：章节内容}
        """
        lines = content.split('\n')
        sections = {}
        current_title = "简介"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是章节标题
            if self._is_section_title(line):
                # 保存上一个章节
                if current_content:
                    sections[current_title] = '\n'.join(current_content)
                
                # 开始新章节
                current_title = line
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_content:
            sections[current_title] = '\n'.join(current_content)
        
        return sections

    def _is_section_title(self, line: str) -> bool:
        """判断是否是章节标题"""
        # 标题通常比较短，且没有句号
        if len(line) > 50:
            return False
        
        # 标题通常以关键词结尾
        title_keywords = [
            '经历', '技能', '教育', '学历', '工作', '项目',
            '简介', '总结', '荣誉', '证书', '语言', '兴趣',
            'Experience', 'Education', 'Skill', 'Project', 'Work'
        ]
        
        return any(keyword in line for keyword in title_keywords)

    def _clean_section(self, content: str) -> str:
        """清理章节内容，去除多余空白"""
        lines = content.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(line)
        return '\n'.join(cleaned)


def parse_resume(file_path: str) -> Dict[str, Optional[str]]:
    """便捷函数，解析简历文件"""
    parser = ResumeParser()
    return parser.parse_file(file_path)


# ---------------------------------------------------------------------------
# 知识库文档切片
# ---------------------------------------------------------------------------

# markdown 标题行（# / ## / ###）作为知识条目的天然边界
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")


@dataclass
class DocumentChunk:
    """知识库切片，入库向量库的最小单元"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]


class DocumentProcessor:
    """知识库文档切片器。

    切片策略：
    - 按 markdown 标题（# ~ ###）切分章节，标题行保留在 chunk 正文中，
      标题文本写入 metadata.section，检索时可展示来源小节
    - 相邻小节贪心合并，单个 chunk 不超过 chunk_size 字
    - 单个章节超长时复用 split_text 按段落/句子边界二次切分
    - 无标题的纯文本直接按 split_text 切块
    """

    SUPPORTED_EXTENSIONS = (".md", ".txt")

    def __init__(self, chunk_size: int = 400):
        self.chunk_size = chunk_size

    def process_directory(self, dir_path: str) -> List[DocumentChunk]:
        """处理目录下全部支持格式的知识文件，按文件名排序保证结果稳定"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            logger.warning(f"知识目录不存在：{dir_path}")
            return []

        chunks: List[DocumentChunk] = []
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                if file_path.is_file():
                    logger.info(f"跳过不支持的文件：{file_path.name}")
                continue
            chunks.extend(self.process_file(str(file_path)))

        logger.info(f"目录切片完成：{dir_path.name} -> {len(chunks)} chunks")
        return chunks

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """处理单个知识文件"""
        from agents.skills.resume_agents.textutil import split_text

        file_path = Path(file_path)
        if not file_path.is_file():
            logger.error(f"知识文件不存在：{file_path}")
            return []

        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            logger.warning(f"知识文件为空：{file_path.name}")
            return []

        stem = file_path.stem
        bodies: List[tuple] = []  # (section_title, chunk_body)
        buffer_title = ""
        buffer_body = ""

        def flush():
            nonlocal buffer_title, buffer_body
            if buffer_body.strip():
                bodies.append((buffer_title, buffer_body.strip()))
            buffer_title, buffer_body = "", ""

        for section_title, section_body in self._split_by_headings(text):
            if len(section_body) > self.chunk_size:
                # 超长章节：先冲刷缓冲，再按段落/句子边界二次切分
                flush()
                for piece in split_text(section_body, self.chunk_size):
                    bodies.append((section_title, piece))
            elif buffer_body and len(buffer_body) + 1 + len(section_body) > self.chunk_size:
                flush()
                buffer_title, buffer_body = section_title, section_body
            else:
                # 相邻小节合并，标题沿用缓冲中第一个非空标题
                if not buffer_body:
                    buffer_title = section_title
                buffer_body = f"{buffer_body}\n{section_body}" if buffer_body else section_body
        flush()

        chunks = [
            DocumentChunk(
                chunk_id=f"{stem}_{i:03d}",
                content=body,
                metadata={
                    "source": file_path.name,
                    "section": title,
                    "chunk_index": i,
                },
            )
            for i, (title, body) in enumerate(bodies)
        ]
        logger.info(f"文件切片完成：{file_path.name} -> {len(chunks)} chunks")
        return chunks

    def _split_by_headings(self, text: str) -> List[tuple]:
        """按 markdown 标题切分为 (标题, 正文) 列表，标题行保留在正文中。

        第一个标题之前的内容归入标题为空的起始章节；全文无标题时整体作为
        一个空标题章节返回。
        """
        sections: List[tuple] = []
        current_title = ""
        current_lines: List[str] = []

        for line in text.split("\n"):
            match = _HEADING_RE.match(line.strip())
            if match:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = match.group(2).strip()
                current_lines = [line.strip()]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return [(title, body) for title, body in sections if body]
