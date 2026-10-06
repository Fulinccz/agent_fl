"""简历润色技能（适配层：内部调用 ResumePolishAgent）"""
from typing import Dict, Any, Optional

from agents.llm import get_shared_llm
from agents.skills.resume_agents.polish_agent import ResumePolishAgent
from logger import get_logger

logger = get_logger(__name__)


def polish_resume(
    content: str,
    jd: Optional[str] = None,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """简历润色技能入口函数

    内部复用 agents.skills.resume_agents.ResumePolishAgent。
    """
    shared_llm = get_shared_llm()
    state: Dict[str, Any] = {
        "resume": content,
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

    polish_agent = ResumePolishAgent(llm=shared_llm)
    state = polish_agent.run(state)

    return {
        "success": state.get("error") is None,
        "polished_content": state.get("optimized_resume", content),
    }
