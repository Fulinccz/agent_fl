"""
Prompt Template Management
集中管理所有 AI 任务的 Prompt 模板
"""

from __future__ import annotations
from typing import Dict, Optional


class ResumePrompts:
    """简历相关任务的 Prompt 模板"""
    
    PARSE_SCHEMA = '''你是专业简历解析师。请严格按照以下JSON格式输出简历的结构化信息：
{
    "success": true,
    "error": null,
    "data": {
        "personal_info": {
            "name": "姓名",
            "phone": "联系电话",
            "email": "邮箱地址",
            "location": "所在城市"
        },
        "education": [
            {
                "school": "学校名称",
                "degree": "学位",
                "major": "专业",
                "start_date": "开始时间",
                "end_date": "结束时间",
                "gpa": "GPA（如有）"
            }
        ],
        "work_experience": [
            {
                "company": "公司名称",
                "position": "职位",
                "start_date": "开始时间",
                "end_date": "结束时间",
                "description": "工作描述"
            }
        ],
        "projects": [
            {
                "name": "项目名称",
                "role": "角色",
                "start_date": "开始时间",
                "end_date": "结束时间",
                "description": "项目描述",
                "tech_stack": ["技术栈"]
            }
        ],
        "skills": {
            "programming_languages": ["编程语言"],
            "frameworks": ["框架"],
            "tools": ["工具"],
            "other_skills": ["其他技能"]
        },
        "certificates": ["证书"],
        "languages": ["语言能力"],
        "summary": "个人总结"
    }
}'''

    SCORE_SCHEMA = '''你是专业HR和职业顾问。请对以下简历进行多维度评分，严格按照JSON格式输出：
{
    "success": true,
    "error": null,
    "scores": {
        "completeness": {
            "score": 0-100,
            "comment": "评价说明",
            "suggestions": ["改进建议"]
        },
        "relevance": {
            "score": 0-100,
            "comment": "与目标职位的匹配度评价",
            "suggestions": ["改进建议"]
        },
        "professionalism": {
            "score": 0-100,
            "comment": "专业性评价",
            "suggestions": ["改进建议"]
        },
        "impact": {
            "score": 0-100,
            "comment": "成果量化程度评价",
            "suggestions": ["改进建议"]
        },
        "structure": {
            "score": 0-100,
            "comment": "简历结构清晰度评价",
            "suggestions": ["改进建议"]
        }
    },
    "overall_score": 0-100,
    "summary": "总体评价",
    "key_strengths": ["主要优势"],
    "critical_weaknesses": ["关键劣势"],
    "priority_improvements": ["优先改进项（按重要度排序）"]
}'''

    OPTIMIZE_SCHEMA = '''你是资深HR专家和职业规划师。请对以下简历进行全面优化分析，严格按照JSON格式输出：
{
    "success": true,
    "task_type": "optimize",
    "error": null,
    "scoring_result": {
        "overall_score": 0-100,
        "dimension_scores": {
            "completeness": 0-100,
            "relevance": 0-100,
            "professionalism": 0-100,
            "achievements": 0-100,
            "formatting": 0-100
        }
    },
    "optimization_result": {
        "improved_summary": "优化后的个人总结/求职意向",
        "improved_experiences": [
            {
                "section": "原始段落标题",
                "original_text": "原始内容",
                "optimized_text": "优化后内容",
                "key_changes": ["主要改动点"],
                "improvement_reason": "优化原因"
            }
        ],
        "full_optimized_resume": "完整的优化后简历文本"
    },
    "matched_skills_from_JD": ["从JD中匹配到的技能关键词"],
    "missing_skills": ["JD要求但简历中缺失的技能"],
    "strengths": ["简历优势"],
    "weaknesses": ["需要改进的方面"],
    "suggestions": [
        {
            "priority": "high/medium/low",
            "category": "类别（content/format/keywords/achievements）",
            "title": "建议标题",
            "description": "详细描述",
            "example": "示例（可选）"
        }
    ],
    "action_items": [
        {
            "action": "具体行动项",
            "impact": "预期影响",
            "effort": "small/medium/large"
        }
    ],
    "final_recommendation": "最终综合建议"
}'''

    @staticmethod
    def get_parse_prompt(resume_content: str) -> str:
        """生成简历解析的完整 Prompt"""
        return f"请解析以下简历内容：\n{resume_content}"

    @staticmethod
    def get_score_prompt(
        resume_content: str,
        job_description: Optional[str] = None
    ) -> str:
        """生成简历评分的完整 Prompt"""
        base = f"请对以下简历进行多维度评分：\n{resume_content}"
        
        if job_description:
            base += f"\n\n目标职位描述：\n{job_description}"
            
        return base

    @staticmethod
    def get_optimize_prompt(
        resume_content: str,
        job_description: Optional[str] = None
    ) -> str:
        """生成简历优化的完整 Prompt"""
        base = f"请全面优化以下简历：\n{resume_content}"
        
        if job_description:
            base += f"\n\n目标职位描述：\n{job_description}"
            
        return base


class TextPrompts:
    """文本处理相关的 Prompt 模板"""

    STYLE_GUIDES = {
        "professional": "专业正式风格，适合简历、报告等正式场合",
        "concise": "简洁明了风格，去除冗余表达，突出重点",
        "persuasive": "有说服力风格，强调成就和价值",
        "academic": "学术风格，严谨客观，适合论文、研究报告"
    }

    POLISH_SCHEMA_TEMPLATE = '''你是专业文案编辑。请对以下文本进行润色，严格按照JSON格式输出：
{{
    "success": true,
    "error": null,
    "original_text": "原文（原样保留）",
    "polished_text": "润色后的文本",
    "style": "{style}",
    "changes": [
        {{
            "type": "修改类型（vocabulary/structure/tone/clarity/conciseness）",
            "original": "原文片段",
            "improved": "修改后片段",
            "reason": "修改原因"
        }}
    ],
    "improvement_summary": "整体改进说明",
    "word_count_change": {{"original": 数字, "polished": 数字}},
    "readability_score": {{
        "before": 0-100,
        "after": 0-100
    }}
}}

要求：保持原文核心意思不变，提升表达的{style_guide}。'''

    @classmethod
    def get_polish_schema(cls, style: str = "professional") -> str:
        """
        获取文本润色的 Schema
        
        Args:
            style: 文本风格
            
        Returns:
            格式化后的 Prompt Schema
        """
        style_guide = cls.STYLE_GUIDES.get(style, cls.STYLE_GUIDES["professional"])
        return cls.POLISH_SCHEMA_TEMPLATE.format(
            style=style,
            style_guide=style_guide
        )

    @staticmethod
    def get_polish_prompt(text: str) -> str:
        """生成文本润色的完整 Prompt"""
        return f"请润色以下文本：\n{text}"


class PromptManager:
    """统一的 Prompt 管理器"""
    
    def __init__(self):
        self._resume = ResumePrompts()
        self._text = TextPrompts()
    
    @property
    def resume(self) -> ResumePrompts:
        """访问简历相关 Prompt"""
        return self._resume
    
    @property
    def text(self) -> TextPrompts:
        """访问文本处理相关 Prompt"""
        return self._text
    
    @staticmethod
    def get_fallback(task_type: str) -> Dict:
        """
        获取指定任务类型的 fallback 响应
        
        Args:
            task_type: 任务类型 ('parse', 'score', 'optimize', 'polish')
            
        Returns:
            标准化的 fallback 字典
        """
        fallbacks = {
            "parse": {
                "success": False,
                "error": "无法解析简历内容",
                "data": {"raw_text": "", "parsed_fields": []}
            },
            "score": {
                "success": False,
                "error": "无法完成评分",
                "scores": {},
                "overall_score": 50,
                "summary": "评分服务暂时不可用",
                "key_strengths": [],
                "critical_weaknesses": [],
                "priority_improvements": []
            },
            "optimize": {
                "success": False,
                "task_type": "optimize",
                "error": "无法完成优化分析",
                "optimization_result": None,
                "scoring_result": {"overall_score": 50},
                "suggestions": [{
                    "priority": "medium",
                    "description": "优化服务暂时不可用，请稍后重试"
                }]
            },
            "polish": {
                "success": False,
                "error": "无法完成润色",
                "polished_text": "",
                "changes": [],
                "style": "professional",
                "improvement_summary": "润色服务暂时不可用"
            }
        }
        
        return fallbacks.get(task_type, {
            "success": False,
            "error": f"未知任务类型: {task_type}"
        })
