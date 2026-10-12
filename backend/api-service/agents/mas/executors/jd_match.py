"""JD 匹配分析执行器"""
from __future__ import annotations

from typing import Any, Dict, Optional

from logger import get_logger
from .base import BaseExecutor, _resolve_llm
from ..state import StepResult

logger = get_logger(__name__)


class JDMatchExecutor(BaseExecutor):
    """JD 匹配分析执行器"""

    name = "jd_match"
    description = "分析简历与目标岗位 JD 的匹配度并生成优化建议"
    required_inputs = ["resume"]  # jd 可选，缺失时产出默认建议

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.match_agent import JDMatchAgent

            agent = JDMatchAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "score_result": task_input.get("score_result") or {
                    "completeness": 70, "professionalism": 70,
                    "quantification": 60, "matching": 70,
                },
                "match_result": None,
                "suggestions": None,
                "overall_score": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            agent.run(state)

            # 兜底错误不写入 result.error（Reflector 会判 RETRY 造成无效重试）；
            # 该 agent 设计上总有产出，错误信息随 data 保留供排查
            if state.get("error"):
                result.data = {"match_fallback_error": state["error"]}

            result.success = True  # 匹配 agent 有完整兜底逻辑，总有产出
            result.data.update({
                "match_result": state.get("match_result"),
                "suggestions": state.get("suggestions"),
            })
            match = state.get("match_result") or {}
            result.summary = f"匹配分析完成: 匹配度{match.get('match_score', '?')}分"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result
