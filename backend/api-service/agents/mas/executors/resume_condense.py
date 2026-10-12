"""简历精简执行器（非流式，条件触发）"""
from __future__ import annotations

from typing import Any, Dict

from logger import get_logger
from .base import BaseExecutor, _resolve_llm
from ..state import StepResult

logger = get_logger(__name__)


class ResumeCondenseExecutor(BaseExecutor):
    """简历精简执行器：超长简历先压缩，再交由润色，保证输出聚焦"""

    name = "resume_condense"
    description = "精简过长简历（>2800 字），压缩至约 60% 篇幅并保留核心信息"
    required_inputs = ["resume"]
    streamable = False

    def __init__(self, llm: Any | None = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.condense_agent import ResumeCondenseAgent

            agent = ResumeCondenseAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "current_step": "started",
            }
            agent.run(state)

            condensed = state.get("condensed_resume")
            if not condensed:
                result.error = state.get("error") or "精简结果为空"
                return result

            result.success = True
            result.data = {
                "condensed_resume": condensed,
                "original_length": state.get("original_length"),
                "condensed_length": state.get("condensed_length"),
            }
            result.summary = (
                f"精简完成: {state.get('original_length', '?')} → "
                f"{state.get('condensed_length', '?')} 字"
            )
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result
