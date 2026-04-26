"""工作流状态定义"""
from typing import Dict, Any, Optional, List, TypedDict


class ResumeState(TypedDict):
    """工作流状态"""
    # 输入
    resume: str
    jd: Optional[str]
    position_type: Optional[str]

    # 中间结果
    score_result: Optional[Dict[str, Any]]
    match_result: Optional[Dict[str, Any]]

    # 输出
    overall_score: Optional[Dict[str, Any]]
    suggestions: Optional[List[Dict[str, Any]]]
    optimized_resume: Optional[str]

    # 控制
    error: Optional[str]
    current_step: str
