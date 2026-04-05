"""
Resume Tools - 简历相关工具集

包含：
- ResumeParserTool: 简历解析工具
- ResumeScorerTool: 简历打分工具
- ResumeOptimizerTool: 简历优化工具

设计原则：
- 每个工具只做一件事（单一职责）
- 继承 BaseTool，使用标准接口
- 可被 LangChain Agent 直接调用
- 支持独立使用或组合使用
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from ..core.base_tool import BaseTool, ToolResult, ToolRegistry
from ..utils.json_utils import JsonUtils
from ..utils.prompts import ResumePrompts, PromptManager


@ToolRegistry.register
class ResumeParserTool(BaseTool):
    """
    简历解析工具
    
    功能：
    - 从非结构化文本中提取简历信息
    - 返回结构化的 JSON 数据
    
    使用示例：
        tool = ResumeParserTool(provider)
        result = tool.execute(resume_content="张三，Python 开发工程师...")
        print(result.data)  # 结构化数据
    """
    
    name = "resume_parser"
    description = "解析简历文本，提取个人信息、教育经历、工作经历、项目经验等结构化信息"
    
    def execute(self, resume_content: str = "", **kwargs) -> ToolResult:
        """
        执行简历解析
        
        Args:
            resume_content: 简历文本内容
            
        Returns:
            ToolResult 包含解析后的结构化数据
        """
        if not self._provider:
            return ToolResult(
                success=False,
                error="未提供模型提供者",
                metadata={"tool": self.name}
            )
        
        if not resume_content.strip():
            return ToolResult(
                success=False,
                error="简历内容为空",
                data={"data": {}},
                metadata={"tool": self.name}
            )
        
        try:
            fallback = PromptManager.get_fallback("parse")
            fallback["data"]["raw_text"] = resume_content
            
            result = JsonUtils.safe_generate(
                generate_func=self._provider.generate,
                prompt=ResumePrompts.get_parse_prompt(resume_content),
                schema_description=ResumePrompts.PARSE_SCHEMA,
                fallback=fallback,
                temperature=0.2,
                max_new_tokens=2048
            )
            
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                metadata={
                    "tool": self.name,
                    "input_length": len(resume_content)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"tool": self.name}
            )


@ToolRegistry.register
class ResumeScorerTool(BaseTool):
    """
    简历打分工具
    
    功能：
    - 多维度评分（完整性、相关性、专业性等）
    - 提供改进建议
    
    使用示例：
        tool = ResumeScorerTool(provider)
        result = tool.execute(
            resume_content="...",
            job_description="..."
        )
        print(result.data["overall_score"])
    """
    
    name = "resume_scorer"
    description = "对简历进行多维度打分，包括完整性、相关性、专业性、成果量化、结构清晰度等维度"
    
    def execute(
        self, 
        resume_content: str = "", 
        job_description: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        执行简历打分
        
        Args:
            resume_content: 简历内容
            job_description: 目标职位描述（可选）
            
        Returns:
            ToolResult 包含评分结果
        """
        if not self._provider:
            return ToolResult(
                success=False,
                error="未提供模型提供者",
                metadata={"tool": self.name}
            )
        
        if not resume_content.strip():
            return ToolResult(
                success=False,
                error="简历内容为空",
                data={"scores": {}, "overall_score": 0},
                metadata={"tool": self.name}
            )
        
        try:
            fallback = PromptManager.get_fallback("score")
            
            result = JsonUtils.safe_generate(
                generate_func=self._provider.generate,
                prompt=ResumePrompts.get_score_prompt(resume_content, job_description),
                schema_description=ResumePrompts.SCORE_SCHEMA,
                fallback=fallback,
                temperature=0.2,
                max_new_tokens=1536
            )
            
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                metadata={
                    "tool": self.name,
                    "has_jd": bool(job_description)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"tool": self.name}
            )


@ToolRegistry.register
class ResumeOptimizerTool(BaseTool):
    """
    简历优化工具
    
    功能：
    - 基于目标职位优化简历内容
    - 提供详细的优化建议和改写
    - 匹配 JD 关键词
    
    使用示例：
        tool = ResumeOptimizerTool(provider)
        result = tool.execute(
            resume_content="...",
            job_description="..."
        )
        print(result.data["optimization_result"]["full_optimized_resume"])
    """
    
    name = "resume_optimizer"
    description = "基于目标职位描述全面优化简历，包括内容改写、关键词匹配、结构优化、亮点突出等"
    
    def execute(
        self, 
        resume_content: str = "", 
        job_description: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        执行简历优化
        
        Args:
            resume_content: 简历内容
            job_description: 目标职位描述（可选）
            
        Returns:
            ToolResult 包含完整的优化分析结果
        """
        if not self._provider:
            return ToolResult(
                success=False,
                error="未提供模型提供者",
                metadata={"tool": self.name}
            )
        
        if not resume_content.strip():
            return ToolResult(
                success=False,
                error="简历内容为空",
                data={
                    "optimization_result": None,
                    "scoring_result": {"overall_score": 50},
                    "suggestions": []
                },
                metadata={"tool": self.name}
            )
        
        try:
            fallback = PromptManager.get_fallback("optimize")
            
            result = JsonUtils.safe_generate(
                generate_func=self._provider.generate,
                prompt=ResumePrompts.get_optimize_prompt(resume_content, job_description),
                schema_description=ResumePrompts.OPTIMIZE_SCHEMA,
                fallback=fallback,
                temperature=0.3,
                max_new_tokens=2048
            )
            
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                metadata={
                    "tool": self.name,
                    "has_jd": bool(job_description)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"tool": self.name}
            )
