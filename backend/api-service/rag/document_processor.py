"""
简历文档解析工具
支持 .docx 和 .pdf 格式，提取技能栏和项目描述
"""

import re
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResumeParser:
    """简历解析器"""
    
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
                logger.warning(f"文件不是真正的 .docx 格式，尝试作为 .doc 处理")
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
            
            logger.info(f"使用 Word COM 接口解析 .doc 文件")
            
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
        
        # 如果没有找到明确的技能章节，尝试从内容中查找包含技术关键词的部分
        lines = content.split('\n')
        skill_lines = []
        in_skill_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是技能相关行
            if any(re.search(keyword, line, re.IGNORECASE) for keyword in self.skill_keywords):
                in_skill_section = True
                continue
            
            # 如果在技能部分，收集内容
            if in_skill_section:
                # 检查是否遇到新的章节标题
                if self._is_section_title(line):
                    break
                skill_lines.append(line)
        
        if skill_lines:
            return '\n'.join(skill_lines)
        
        # 如果还是没找到，返回前 30% 的内容作为技能栏
        lines = content.split('\n')
        skill_lines = lines[:max(1, len(lines) // 3)]
        return '\n'.join(skill_lines)

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
