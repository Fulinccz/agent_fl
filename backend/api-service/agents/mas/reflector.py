"""反思器（Reflector）：步骤结果的评审裁决层

职责边界（与 Master 控制策略分离）：
- 只做"评估"：给定 (步骤, 结果, 目标, 轨迹)，产出 Reflection(verdict/feedback)
  或 None（无法裁决）；不持有重试预算、不做重试/重规划控制决策
- 控制策略（重试预算、重规划、跳过）统一由 MasterOrchestrator._handle_verdict 执行

评审链：RuleCritic（确定性规则，零 LLM）→ LLMCritic（边界情况深审）→ None
- 裁决词汇：continue / retry / replan / finish / abort
- retry 仅表示"需要带反馈再试一次"，是否重试由 Master 按预算决定
"""
from __future__ import annotations

import json
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

# 润色产出最小有效长度：低于此值视为无意义输出，升级 LLM 审查
MIN_POLISH_LENGTH = 30


class RuleCritic:
    """规则评审器：确定性快评，零 LLM 开销

    返回 None 表示属于边界情况（需要 LLM 深审）。
    只报告"结果是否有问题"，不关心重试次数。
    """

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

    def critique(self, step: TaskStep, result: StepResult) -> Optional[Reflection]:
        """确定性规则裁决；返回 None 表示边界情况，交由 LLM 深审"""
        # 执行出错（重试预算由 Master 控制，这里只报告问题）
        if not result.success or result.error:
            return Reflection(
                step_id=step.step_id, verdict=VERDICT_RETRY,
                thought=f"执行失败: {result.error[:120]}",
                feedback=f"上次失败原因: {result.error[:200]}，请修正后重试",
                source="rule",
            )

        # 按执行器类型校验关键产出（硬缺失）
        data = result.data or {}
        missing = self._check_outputs(step.agent, data)
        if missing:
            return Reflection(
                step_id=step.step_id, verdict=VERDICT_RETRY,
                thought=f"产出缺失关键字段: {missing}",
                feedback=f"缺少 {missing}，请确保完整生成",
                source="rule",
            )

        # 边界情况：润色产出过短（非空但可能无意义）→ 交 LLM 判断是否可接受
        if step.agent == "resume_polish":
            polished = (data.get("optimized_resume") or "").strip()
            if len(polished) < MIN_POLISH_LENGTH:
                logger.info(
                    f"[RuleCritic] 步骤 {step.step_id} 润色产出过短({len(polished)}字)，升级 LLM 审查"
                )
                return None

        # 明确通过
        return Reflection(
            step_id=step.step_id, verdict=VERDICT_CONTINUE,
            thought=f"步骤 {step.step_id} 执行成功，{result.summary}",
            source="rule",
        )


class LLMCritic:
    """LLM 评审器：边界情况的深入审查（Thought → Verdict）"""

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def critique(
        self,
        step: TaskStep,
        result: StepResult,
        goal: str,
        trace: List[str],
    ) -> Optional[Reflection]:
        """调用 LLM 做裁决审查；解析失败返回 None"""
        if self._llm is None:
            try:
                self._llm = get_shared_llm()
            except Exception as e:
                logger.warning(f"[LLMCritic] LLM 不可用: {e}")
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
            logger.warning(f"[LLMCritic] LLM 审查失败: {e}")
            return None

    def _parse_verdict(self, raw: str, step_id: str) -> Optional[Reflection]:
        from services.json_utils import extract_json_object
        json_str = extract_json_object(raw)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
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


class Reflector:
    """反思器：评审链编排（规则快评 → LLM 深审），纯评估不做控制决策

    review 返回 None 表示两位评审均无法裁决（如 LLM 关闭且规则遇边界情况），
    由 Master 按自身策略保守处理。
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        use_llm_reflection: bool = True,
    ):
        self.rule_critic = RuleCritic()
        self.llm_critic = LLMCritic(llm=llm)
        self.use_llm_reflection = use_llm_reflection

    def review(
        self,
        step: TaskStep,
        result: StepResult,
        goal: str,
        trace: List[str],
    ) -> Optional[Reflection]:
        """审查单步执行结果；返回 None 表示无法裁决（Master 兜底）"""
        # ---------- 规则快评（确定性质检） ----------
        rule_reflection = self.rule_critic.critique(step, result)
        if rule_reflection is not None:
            return rule_reflection

        # ---------- 边界情况：LLM 深入审查 ----------
        if self.use_llm_reflection:
            return self.llm_critic.critique(step, result, goal, trace)

        return None
