"""ReAct 反思器（Reflector）

对每一步执行结果进行 ReAct 审查（Thought → Verdict）：
- 规则快评：成功且产出齐全 → continue（零 LLM 开销）
- 失败/产出异常 → 规则裁决重试或升级 LLM 审查
- 裁决：continue / retry / replan / finish / abort
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from logger import get_logger
from services.tracing import start_span, update_current_generation
from agents.llm import get_shared_llm
from .state import (
    Reflection,
    StepResult,
    TaskStep,
    VERDICT_CONTINUE,
    VERDICT_RETRY,
    VERDICT_REPLAN,
    VERDICT_FINISH,
    VERDICT_ABORT,
)
from .prompts import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_PROMPT

logger = get_logger(__name__)


class Reflector:
    """ReAct 反思器"""

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_retries_per_step: int = 2,
        use_llm_reflection: bool = True,
    ):
        self._llm = llm
        self.max_retries_per_step = max_retries_per_step
        self.use_llm_reflection = use_llm_reflection

    def review(
        self,
        step: TaskStep,
        result: StepResult,
        goal: str,
        trace: List[str],
    ) -> Reflection:
        """审查单步执行结果，给出 ReAct 裁决

        决策顺序：规则快评（确定性质检）→ LLM 审查（边界情况）→ 保守兜底。
        """
        # ---------- 规则快评（明确合格/明确不合格直接裁决） ----------
        rule_reflection = self._rule_review(step, result)
        if rule_reflection is not None:
            return rule_reflection

        # ---------- 边界情况：LLM 深入审查 ----------
        if self.use_llm_reflection:
            llm_reflection = self._llm_review(step, result, goal, trace)
            if llm_reflection is not None:
                return llm_reflection

        # LLM 审查失败 → 保守裁决：重试或终止
        if step.retries < self.max_retries_per_step:
            return Reflection(
                step_id=step.step_id, verdict=VERDICT_RETRY,
                thought="结果边界异常且 LLM 审查不可用，保守重试",
                feedback="请检查输出格式后重新执行", source="fallback",
            )
        return Reflection(
            step_id=step.step_id, verdict=VERDICT_REPLAN,
            thought="多次重试仍异常，需要重新规划", source="fallback",
        )

    # ---------- 规则快评 ----------

    def _rule_review(self, step: TaskStep, result: StepResult) -> Optional[Reflection]:
        """确定性规则裁决；返回 None 表示属于边界情况，需要 LLM 深入审查"""
        # 执行出错
        if not result.success or result.error:
            if step.retries < self.max_retries_per_step:
                return Reflection(
                    step_id=step.step_id, verdict=VERDICT_RETRY,
                    thought=f"执行失败: {result.error[:120]}，带着反馈重试",
                    feedback=f"上次失败原因: {result.error[:200]}，请修正后重试",
                    source="rule",
                )
            return Reflection(
                step_id=step.step_id, verdict=VERDICT_REPLAN,
                thought=f"步骤 {step.step_id} 重试 {step.retries} 次仍失败，触发重规划",
                source="rule",
            )

        # 按执行器类型校验关键产出（硬缺失）
        data = result.data or {}
        missing = self._check_outputs(step.agent, data)

        if missing:
            if step.retries < self.max_retries_per_step:
                return Reflection(
                    step_id=step.step_id, verdict=VERDICT_RETRY,
                    thought=f"产出缺失关键字段: {missing}，重试",
                    feedback=f"缺少 {missing}，请确保完整生成",
                    source="rule",
                )
            return Reflection(
                step_id=step.step_id, verdict=VERDICT_REPLAN,
                thought="产出始终缺失关键字段，触发重规划", source="rule",
            )

        # 边界情况：润色产出过短（非空但可能无意义）→ 交 LLM 判断是否可接受
        if step.agent == "resume_polish":
            polished = (data.get("optimized_resume") or "").strip()
            if len(polished) < 30:
                logger.info(f"[Reflector] 步骤 {step.step_id} 润色产出过短({len(polished)}字)，升级 LLM 审查")
                return None

        # 明确通过
        return Reflection(
            step_id=step.step_id, verdict=VERDICT_CONTINUE,
            thought=f"步骤 {step.step_id} 执行成功，{result.summary}",
            source="rule",
        )

    @staticmethod
    def _check_outputs(agent: str, data: Dict[str, Any]) -> List[str]:
        """校验各执行器必须产出的字段，返回缺失字段名（仅做硬性存在性校验）"""
        required: Dict[str, List[str]] = {
            "resume_score": ["score_result", "overall_score"],
            "jd_match": ["match_result", "suggestions"],
            "resume_polish": ["optimized_resume"],
            "resume_parse": [],
            "chat": ["answer"],
        }
        keys = required.get(agent, [])
        missing = []
        for key in keys:
            value = data.get(key)
            if value is None or value == "" or value == []:
                missing.append(key)
        return missing

    # ---------- LLM 审查 ----------

    def _llm_review(
        self,
        step: TaskStep,
        result: StepResult,
        goal: str,
        trace: List[str],
    ) -> Optional[Reflection]:
        """调用 LLM 做 ReAct 审查；解析失败返回 None"""
        if self._llm is None:
            try:
                self._llm = get_shared_llm()
            except Exception as e:
                logger.warning(f"[Reflector] LLM 不可用: {e}")
                return None

        observation = json.dumps(result.to_dict(), ensure_ascii=False)[:800]
        prompt = (
            f"{REFLECTOR_SYSTEM_PROMPT}\n\n"
            + REFLECTOR_USER_PROMPT.format(
                goal=goal[:300],
                step_id=step.step_id,
                agent=step.agent,
                task=step.task[:200],
                observation=observation,
                trace="\n".join(trace[-4:]) or "无",
            )
        )

        try:
            with start_span("llm-reflect", as_type="generation", input=prompt[:500]):
                raw = self._llm.generate(prompt, deepThinking=False, max_new_tokens=160)
                update_current_generation(output=raw[:300])
            return self._parse_verdict(raw, step.step_id)
        except Exception as e:
            logger.warning(f"[Reflector] LLM 审查失败: {e}")
            return None

    def _parse_verdict(self, raw: str, step_id: str) -> Optional[Reflection]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        verdict = str(data.get("verdict", "")).strip().lower()
        valid = {VERDICT_CONTINUE, VERDICT_RETRY, VERDICT_REPLAN,
                 VERDICT_FINISH, VERDICT_ABORT}
        if verdict not in valid:
            return None

        return Reflection(
            step_id=step_id,
            verdict=verdict,
            thought=str(data.get("thought", ""))[:200],
            feedback=str(data.get("feedback", ""))[:200],
            source="llm",
        )
