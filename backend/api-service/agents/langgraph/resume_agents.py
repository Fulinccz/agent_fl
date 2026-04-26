"""
简历优化多 Agent 工作流
基于 LangGraph 实现 3 个 Agent 的协作

注意：此文件为兼容性入口，实际实现已拆分到 resume_agents/ 目录下
"""

# 从新的模块位置导入所有内容，保持向后兼容
from .resume_agents.state import ResumeState
from .resume_agents.score_agent import ResumeScoreAgent
from .resume_agents.match_agent import JDMatchAgent
from .resume_agents.polish_agent import ResumePolishAgent
from .resume_agents.workflow import (
    ResumeOptimizationWorkflow,
    optimize_stream,
    get_resume_workflow,
    create_resume_workflow,
)

__all__ = [
    'ResumeState',
    'ResumeScoreAgent',
    'JDMatchAgent',
    'ResumePolishAgent',
    'ResumeOptimizationWorkflow',
    'optimize_stream',
    'get_resume_workflow',
    'create_resume_workflow',
]
