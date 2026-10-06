"""JD 关键词匹配技能（适配层：内部调用 JDMatchAgent）"""
from typing import Dict, Any

from agents.llm import get_shared_llm
from agents.skills.resume_agents.match_agent import JDMatchAgent
from logger import get_logger

logger = get_logger(__name__)


def match_jd_keywords(
    resume: str,
    jd: str,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """JD 关键词匹配技能入口函数

    内部复用 agents.skills.resume_agents.JDMatchAgent。
    """
    shared_llm = get_shared_llm()
    state: Dict[str, Any] = {
        "resume": resume,
        "jd": jd,
        "position_type": None,
        "score_result": None,
        "match_result": None,
        "overall_score": None,
        "suggestions": None,
        "optimized_resume": None,
        "error": None,
        "current_step": "started",
    }

    match_agent = JDMatchAgent(llm=shared_llm)
    state = match_agent.run(state)

    match_result = state.get("match_result") or {}

    return {
        "success": state.get("error") is None,
        "match_analysis": match_result,
        "extracted_keywords": {
            "matched_keywords": match_result.get("matched_keywords", []),
            "missing_keywords": match_result.get("missing_keywords", []),
        },
        "optimized_resume": None,
    }
