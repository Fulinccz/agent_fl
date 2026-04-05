"""
Resume Service
简历相关的业务服务（解析、优化、打分）

职责：
- 简历解析：提取结构化信息
- 简历优化：基于 JD 优化内容
- 简历打分：多维度评分

设计原则：
- 单一职责：只处理简历相关业务
- 依赖注入：通过 Agent 实例调用模型
- 统一输出：所有方法返回标准 JSON 格式
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from ..utils.json_utils import JsonUtils
from ..utils.prompts import ResumePrompts, PromptManager


class ResumeService:
    """简历服务类 - 处理所有简历相关任务"""
    
    def __init__(self, agent):
        """
        初始化简历服务
        
        Args:
            agent: Agent 实例（LocalAgent 或 OnlineAgent）
        """
        self.agent = agent
        self.prompts = ResumePrompts()
    
    def parse(self, resume_content: str) -> Dict[str, Any]:
        """
        解析简历，提取结构化信息
        
        Args:
            resume_content: 简历文本内容
            
        Returns:
            包含结构化信息的字典：
            {
                "success": bool,
                "error": str | None,
                "data": {
                    "personal_info": {...},
                    "education": [...],
                    "work_experience": [...],
                    ...
                }
            }
        """
        if not resume_content.strip():
            return JsonUtils.create_error_response(
                task_type="parse",
                error_message="简历内容为空",
                additional_data={"data": {}}
            )
        
        fallback = PromptManager.get_fallback("parse")
        fallback["data"]["raw_text"] = resume_content
        
        return JsonUtils.safe_generate(
            generate_func=self.agent.generate,
            prompt=self.prompts.get_parse_prompt(resume_content),
            schema_description=self.prompts.PARSE_SCHEMA,
            fallback=fallback,
            temperature=0.2,
            max_new_tokens=2048
        )
    
    def score(
        self,
        resume_content: str,
        job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对简历进行多维度评分
        
        Args:
            resume_content: 简历文本内容
            job_description: 目标职位描述（可选）
            
        Returns:
            包含评分结果的字典：
            {
                "success": bool,
                "scores": {...},
                "overall_score": int (0-100),
                "summary": str,
                ...
            }
        """
        if not resume_content.strip():
            return JsonUtils.create_error_response(
                task_type="score",
                error_message="简历内容为空",
                additional_data={
                    "scores": {},
                    "overall_score": 0
                }
            )
        
        fallback = PromptManager.get_fallback("score")
        
        return JsonUtils.safe_generate(
            generate_func=self.agent.generate,
            prompt=self.prompts.get_score_prompt(resume_content, job_description),
            schema_description=self.prompts.SCORE_SCHEMA,
            fallback=fallback,
            temperature=0.2,
            max_new_tokens=1536
        )
    
    def optimize(
        self,
        resume_content: str,
        job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        全面优化简历内容
        
        Args:
            resume_content: 简历文本内容
            job_description: 目标职位描述（可选）
            
        Returns:
            包含完整优化分析的字典：
            {
                "success": bool,
                "task_type": "optimize",
                "scoring_result": {...},
                "optimization_result": {...},
                "suggestions": [...],
                ...
            }
        """
        if not resume_content.strip():
            return JsonUtils.create_error_response(
                task_type="optimize",
                error_message="简历内容为空",
                additional_data={
                    "optimization_result": None,
                    "scoring_result": None,
                    "suggestions": []
                }
            )
        
        fallback = PromptManager.get_fallback("optimize")
        
        result = JsonUtils.safe_generate(
            generate_func=self.agent.generate,
            prompt=self.prompts.get_optimize_prompt(resume_content, job_description),
            schema_description=self.prompts.OPTIMIZE_SCHEMA,
            fallback=fallback,
            temperature=0.3,
            max_new_tokens=2048
        )
        
        return result
    
    def full_analysis(
        self,
        resume_content: str,
        job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        完整的简历分析（解析 + 打分 + 优化）
        
        Args:
            resume_content: 简历文本内容
            job_description: 目标职位描述（可选）
            
        Returns:
            包含所有分析结果的字典
        """
        parsed = self.parse(resume_content)
        scored = self.score(resume_content, job_description)
        optimized = self.optimize(resume_content, job_description)
        
        return {
            "success": True,
            "task_type": "full_analysis",
            "parsed_resume": parsed,
            "scoring_result": scored,
            "optimization_result": optimized,
            "summary": {
                "overall_score": scored.get("overall_score", 0),
                "key_strengths": optimized.get("strengths", []),
                "priority_actions": optimized.get("action_items", [])[:5]
            }
        }
