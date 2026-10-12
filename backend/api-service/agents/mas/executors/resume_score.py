"""简历评分执行器"""
from __future__ import annotations

from typing import Any, Dict, Optional

from logger import get_logger
from .base import BaseExecutor, _resolve_llm
from ..state import StepResult

logger = get_logger(__name__)


class ResumeScoreExecutor(BaseExecutor):
    """简历评分执行器"""

    name = "resume_score"
    description = "对简历进行多维度评分（完整性/专业度/量化程度/匹配度）"
    required_inputs = ["resume"]

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.score_agent import ResumeScoreAgent

            agent = ResumeScoreAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "score_result": None,
                "overall_score": None,
                "suggestions": None,
                "match_result": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            agent.run(state)

            if state.get("error"):
                result.error = state["error"]
                # 评分 agent 失败时会写入兜底分数，仍视为部分产出
                result.data = {
                    "score_result": state.get("score_result"),
                    "overall_score": state.get("overall_score"),
                    "dimension_analysis": state.get("dimension_analysis") or {},
                }
                result.summary = "评分执行出错，已使用兜底分数"
                return result

            result.success = True
            result.data = {
                "score_result": state.get("score_result"),
                "overall_score": state.get("overall_score"),
                "dimension_analysis": state.get("dimension_analysis") or {},
            }
            overall = state.get("overall_score") or {}
            result.summary = f"评分完成: {overall.get('score', '?')}分 ({overall.get('rating', '')})"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result
