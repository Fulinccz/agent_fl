"""
简历润色技能实现
将大白话转换为专业简历话术 - 分段处理版本
"""

from typing import Dict, Any, Optional
from agents.registry import get_agent
from rag.retriever import RAGRetriever
from logger import get_logger
import re

logger = get_logger(__name__)


class ResumePolishingSkill:
    """简历润色技能"""
    
    def __init__(self):
        self.agent = get_agent(provider="local")
        self.retriever = RAGRetriever()
    
    def execute(
        self,
        content: str,
        jd: Optional[str] = None,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        执行简历润色 - 分段处理
        
        Args:
            content: 原始简历内容
            jd: 目标岗位描述（可选）
            use_rag: 是否使用 RAG 检索行业动词库
        
        Returns:
            润色后的简历
        """
        # 解析内容，分离技能和项目
        skills_content, projects_content = self._parse_content(content)
        
        # 分别润色技能和项目
        polished_skills = self._polish_skills(skills_content, jd)
        polished_projects = self._polish_projects(projects_content, jd)
        
        # 合并结果
        polished_content = f"""【技能】
{polished_skills}

【项目经历】
{polished_projects}"""
        
        return {
            "success": True,
            "polished_content": polished_content,
        }
    
    def _parse_content(self, content: str) -> tuple:
        """解析内容，分离技能和项目"""
        skills = ""
        projects = ""
        
        # 尝试匹配【技能】和【项目经历】标记
        skill_match = re.search(r'【技能】\s*(.*?)\s*(?=【项目经历】|$)', content, re.DOTALL)
        project_match = re.search(r'【项目经历】\s*(.*?)\s*$', content, re.DOTALL)
        
        if skill_match:
            skills = skill_match.group(1).strip()
        if project_match:
            projects = project_match.group(1).strip()
        
        # 如果没找到标记，尝试其他方式
        if not skills and not projects:
            # 按段落分割，假设前半部分是技能，后半部分是项目
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if len(paragraphs) >= 2:
                skills = paragraphs[0]
                projects = '\n\n'.join(paragraphs[1:])
            else:
                skills = content
        
        return skills, projects
    
    def _polish_skills(self, skills_content: str, jd: Optional[str]) -> str:
        """润色技能部分 - 逐条处理"""
        if not skills_content or skills_content == '无':
            return "无"
        
        # 将技能按行分割
        skill_lines = [line.strip() for line in skills_content.split('\n') if line.strip()]
        
        polished_lines = []
        for line in skill_lines:
            # 跳过空行和标记
            if not line or line in ['【技能】', '技能：']:
                continue
            
            # 单条技能润色
            prompt = self._build_single_skill_prompt(line, jd)
            result = self.agent.generate(prompt, deepThinking=False)
            polished = self._clean_single_result(result)
            
            if polished and len(polished) > 10:
                polished_lines.append(polished)
            else:
                # 如果润色失败，保留原内容
                polished_lines.append(line)
        
        return '\n'.join(polished_lines) if polished_lines else skills_content
    
    def _polish_projects(self, projects_content: str, jd: Optional[str]) -> str:
        """润色项目经历部分 - 逐项目处理"""
        if not projects_content or projects_content == '无':
            return "无"
        
        # 将项目按段落分割（假设每个项目之间有空行）
        project_paragraphs = [p.strip() for p in projects_content.split('\n\n') if p.strip()]
        
        polished_projects = []
        for paragraph in project_paragraphs:
            if not paragraph:
                continue
            
            # 单个项目润色
            prompt = self._build_single_project_prompt(paragraph, jd)
            result = self.agent.generate(prompt, deepThinking=False)
            polished = self._clean_single_result(result)
            
            if polished and len(polished) > 20:
                polished_projects.append(polished)
            else:
                # 如果润色失败，保留原内容
                polished_projects.append(paragraph)
        
        return '\n\n'.join(polished_projects) if polished_projects else projects_content
    
    def _build_single_skill_prompt(self, skill: str, jd: Optional[str]) -> str:
        """构建单条技能的润色 prompt - 简化版"""
        jd_hint = f"目标岗位：{jd}\n" if jd else ""
        
        return f"""{jd_hint}改写以下技能描述，使用专业动词和量化数据：

{skill}

直接输出改写结果，不要解释："""
    
    def _build_single_project_prompt(self, project: str, jd: Optional[str]) -> str:
        """构建单个项目的润色 prompt - 简化版"""
        jd_hint = f"目标岗位：{jd}\n" if jd else ""
        
        return f"""{jd_hint}改写以下项目经历，突出技术难点和个人贡献：

{project}

直接输出改写结果，不要解释："""
    
    def _clean_single_result(self, result: str) -> str:
        """清理单条润色结果 - 强制提取改写后的内容"""
        import json
        
        # 移除思考标签
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除各种代码块和标签
        result = re.sub(r'<ref>.*?</ref>', '', result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(r'<tool>.*?</tool>', '', result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(r'```[\s\S]*?```', '', result)
        result = re.sub(r'`[^`]*`', '', result)
        
        # 尝试找到标记之后的内容
        markers = [
            '直接输出改写结果，不要解释：',
            '改写结果：',
            '改写：',
            '输出：'
        ]
        
        for marker in markers:
            if marker in result:
                idx = result.rfind(marker)
                if idx != -1:
                    after_marker = result[idx + len(marker):].strip()
                    # 清理残留
                    after_marker = re.sub(r'<think>.*', '', after_marker, flags=re.DOTALL | re.IGNORECASE)
                    after_marker = re.sub(r'```[\s\S]*', '', after_marker)
                    # 检查是否是有效内容（不是JSON、不是列表标记）
                    if len(after_marker) > 10 and not after_marker.strip().startswith(('{', '[', '<')):
                        return after_marker.strip()
        
        # 逐行过滤策略
        skip_patterns = [
            '请将以下', '改写为专业表述', '要求：', '示例：', '原文：', '请改写',
            '使用专业动词', '添加量化数据', '去除口语', '只输出改写',
            '目标岗位：', '技能描述', '项目经历',
            'assistant', 'Assistant', 'ASSISTANT', 'user', 'User',
            'Thinking Process:', 'Analyze', 'Input text:', 'Output:',
            '<ref>', '</ref>', '<tool>', '</tool>', '```', 
            '- [ ]', '- [x]', '- []', '[ ]', '[x]',
            '"description"', '"skills"', '"accuracy"',
        ]
        
        lines = result.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 跳过包含跳过模式的行
            should_skip = False
            for pattern in skip_patterns:
                if pattern in stripped:
                    should_skip = True
                    break
            if should_skip:
                continue
            
            # 跳过 JSON 格式的行
            if stripped.startswith('{') or stripped.startswith('['):
                continue
            if stripped in ['}', ']', '},', '],']:
                continue
            
            # 跳过工具调用格式的行
            if stripped.startswith('<') and stripped.endswith('>'):
                continue
            
            # 跳过纯列表标记行
            if re.match(r'^[-*•]\s*$', stripped):
                continue
            
            # 跳过数字列表标记（单独的数字）
            if re.match(r'^\d+[.、]\s*$', stripped):
                continue
            
            # 保留看起来是正常文本的行
            if len(stripped) > 5:
                cleaned_lines.append(stripped)
        
        # 如果清理后内容太少，尝试提取最长的一段文本
        if len(cleaned_lines) == 0:
            # 找最长的一段连续文本
            paragraphs = result.split('\n\n')
            longest = ""
            for p in paragraphs:
                p_clean = p.strip()
                # 跳过明显不是内容的段落
                if p_clean.startswith(('{', '[', '<', '- [', '1.', '2.', '3.')):
                    continue
                if len(p_clean) > len(longest) and len(p_clean) > 10:
                    longest = p_clean
            return longest if longest else "改写失败"
        
        return '\n'.join(cleaned_lines)
    
    def _build_prompt(self, content: str, jd: Optional[str]) -> str:
        """构建润色 prompt（保留用于兼容）"""
        
        prompt = f"""请将以下简历内容改写为专业表述。

{content}

改写要求：
1. 必须改写，不能复制原文
2. 使用专业动词：主导、设计、实现、优化、重构
3. 添加量化数据：提升30%、处理10万+、服务50+用户等
4. 去除口语化表达
5. 保持【技能】和【项目经历】的标记结构
6. 输出完整的改写内容，不要截断
7. 不要输出任何解释或思考过程

示例：
原文：用Python写了一个数据处理脚本
改写：基于Python开发自动化数据处理引擎，实现10万+条记录的清洗与转换，效率提升80%

请改写："""
        
        if jd:
            prompt = f"""目标岗位：{jd}

{prompt}"""
        
        return prompt
    
    def _simple_clean(self, result: str) -> str:
        """清理模型输出 - 强制提取改写后的内容（保留用于兼容）"""
        # 移除思考标签
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)
        
        logger.debug(f"清理前内容长度: {len(result)}")
        
        # 策略：找到最后一个 "请改写：" 或类似标记之后的内容
        # 或者找到 **专业技能** / 【技能】标记之后的内容
        
        # 尝试找到关键标记之后的内容
        markers = [
            '请改写：',
            '**专业技能**',
            '* **专业技能**',
            '【技能】',
            '改写后的简历：',
            '优化后的简历：'
        ]
        
        for marker in markers:
            if marker in result:
                # 找到最后一个出现的位置
                idx = result.rfind(marker)
                if idx != -1:
                    # 提取标记之后的内容
                    after_marker = result[idx + len(marker):].strip()
                    logger.debug(f"找到标记 '{marker}'，之后内容长度: {len(after_marker)}")
                    if len(after_marker) > 50:  # 确保有足够内容
                        return after_marker
        
        # 如果没找到标记，尝试找到第一个以 ** 或 【 开头的行
        lines = result.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 找到技能相关的标记行（支持多种格式）
            skill_markers = ['**专业技能**', '* **专业技能**', '【技能】', '**技能**', '**Skills**', '技能：', 'Skills:']
            if any(stripped == marker or stripped.startswith(marker) for marker in skill_markers):
                # 返回从这一行开始的所有内容
                content = '\n'.join(lines[i:]).strip()
                logger.debug(f"找到技能标记 '{stripped}'，内容长度: {len(content)}")
                return content
        
        # 兜底：移除所有指令行，保留可能是简历内容的行
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 跳过明显的指令行
            skip_patterns = [
                '你是', '改写要求', '示例', '原文', '请直接输出',
                '必须改写', '使用专业动词', '添加量化', '去除口语',
                '保持', '只输出', '目标岗位',
                '1.', '2.', '3.', '4.', '5.', '6.',
                '喜欢不断学习', '热爱计算机', '自我评价'
            ]
            if any(pattern in stripped for pattern in skip_patterns):
                continue
            
            # 跳过 assistant 标记
            if stripped in ['assistant', 'Assistant', 'ASSISTANT']:
                continue
            
            # 保留可能是简历内容的行（较长的行或包含技术关键词）
            if len(stripped) > 20 or any(kw in stripped for kw in ['Python', 'Java', '开发', '设计', '实现']):
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        logger.debug(f"兜底处理后内容长度: {len(result)}")
        return result.strip() if result.strip() else "改写失败，请重试。"


def polish_resume(
    content: str,
    jd: Optional[str] = None,
    use_rag: bool = True
) -> Dict[str, Any]:
    """简历润色技能入口函数"""
    skill = ResumePolishingSkill()
    return skill.execute(content, jd, use_rag)
