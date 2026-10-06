"""简历评分技能（适配层：内部调用 ResumeScoreAgent + JDMatchAgent）"""
from typing import Dict, Any, Optional

from agents.llm import get_shared_llm
from agents.skills.resume_agents.score_agent import ResumeScoreAgent
from agents.skills.resume_agents.match_agent import JDMatchAgent
from logger import get_logger

logger = get_logger(__name__)


def score_resume(
    resume: str,
    jd: Optional[str] = None,
    position_type: Optional[str] = None,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """简历评分技能入口函数

    内部复用 agents.skills.resume_agents 的 Agent，消除重复实现。
    """
    shared_llm = get_shared_llm()
    state: Dict[str, Any] = {
        "resume": resume,
        "jd": jd,
        "position_type": position_type,
        "score_result": None,
        "match_result": None,
        "overall_score": None,
        "suggestions": None,
        "optimized_resume": None,
        "error": None,
        "current_step": "started",
    }

    score_agent = ResumeScoreAgent(llm=shared_llm)
    state = score_agent.run(state)

    match_agent = JDMatchAgent(llm=shared_llm)
    state = match_agent.run(state)

    return {
        "success": state.get("error") is None,
        "overall_score": state.get("overall_score"),
        "scores": state.get("score_result"),
        "suggestions": state.get("suggestions", []),
    }
