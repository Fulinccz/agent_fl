"""简历润色执行器（支持流式）"""
from __future__ import annotations

from typing import Any, Dict, Generator, Optional

from logger import get_logger
from .base import BaseExecutor, _resolve_llm
from ..state import StepResult

logger = get_logger(__name__)


class ResumePolishExecutor(BaseExecutor):
    """简历润色执行器（支持流式）"""

    name = "resume_polish"
    description = "基于评分与建议对简历表述进行专业化润色改写"
    required_inputs = ["resume"]
    streamable = True

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.polish_agent import ResumePolishAgent

            agent = ResumePolishAgent(llm=_resolve_llm(self._llm, task_input))
            # 超长简历优先使用精简稿（由 resume_condense 步骤产出）
            resume = task_input.get("condensed_resume") or task_input["resume"]
            state = {
                "resume": resume,
                "jd": task_input.get("jd"),
                # 上传链路用户的补充要求
                "extra_instruction": task_input.get("extra_instruction"),
                "score_result": None,
                "match_result": None,
                "suggestions": None,
                "overall_score": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            agent.run(state)

            polished = state.get("optimized_resume")
            if not polished:
                result.error = state.get("error") or "润色结果为空"
                return result

            result.success = True
            result.data = {"optimized_resume": polished}
            result.summary = f"润色完成，输出 {len(polished)} 字"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result

    def run_stream(self, task_input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """流式润色：逐块 yield {"type": "token"/"error", ...}"""
        error = self.validate(task_input)
        if error:
            yield {"type": "error", "content": error}
            return

        try:
            from agents.skills.resume_agents.polish_agent import ResumePolishAgent

            agent = ResumePolishAgent(llm=_resolve_llm(self._llm, task_input))
            # 超长简历优先使用精简稿（由 resume_condense 步骤产出）
            resume = task_input.get("condensed_resume") or task_input["resume"]
            state = {
                "resume": resume,
                "jd": task_input.get("jd"),
                # 上传链路用户的补充要求
                "extra_instruction": task_input.get("extra_instruction"),
                "score_result": None,
                "match_result": None,
                "suggestions": None,
                "overall_score": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            yield from agent.run_stream(state)
        except Exception as e:
            logger.error(f"[{self.name}] 流式执行失败: {e}")
            yield {"type": "error", "content": str(e)}

    def finalize(self, raw_text: str, original_resume: str = "", task_input: Optional[Dict[str, Any]] = None) -> str:
        """对累积的流式文本做最终清理与排版（复用润色 agent 的清理逻辑 + 通用格式化）"""
        from agents.skills.resume_agents.polish_agent import ResumePolishAgent

        agent = ResumePolishAgent(llm=_resolve_llm(self._llm, task_input or {}))
        cleaned = agent._clean_result(raw_text, original_resume)
        formatted = agent._format_output(cleaned or raw_text.strip())
        return formatted or cleaned or raw_text.strip()
