"""Master Agent：任务规划器

职责：
1. plan()      依据意图产出任务计划（优先模板，复杂任务用 LLM 分解）
2. replan()    执行受阻时重新规划剩余步骤
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from logger import get_logger
from services.tracing import start_span, update_current_generation
from agents.llm import get_shared_llm
from .state import (
    TaskPlan,
    TaskStep,
    IntentResult,
    INTENT_RESUME_OPTIMIZE,
    INTENT_RESUME_SCORE,
    INTENT_RESUME_POLISH,
    INTENT_JD_MATCH,
    INTENT_RESUME_PARSE,
    INTENT_COMPLEX_TASK,
)
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REPLANNER_USER_PROMPT,
)

logger = get_logger(__name__)

MAX_LLM_STEPS = 4


class MasterAgent:
    """Master Agent：负责任务分解与计划编排（不直接执行）"""

    def __init__(self, llm: Optional[Any] = None, executor_catalog: str = ""):
        self._llm = llm
        self.executor_catalog = executor_catalog

    # ---------- 对外接口 ----------

    def plan(
        self,
        user_input: str,
        intent: IntentResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """产出任务计划

        简单意图，以及简历优化等自带确定性模板的复杂意图 → 模板（零 LLM 开销）
        通用复合任务（complex-task，无确定性模板）→ 先 LLM 分解，失败再回退模板
        """
        if intent.is_complex and intent.name == INTENT_COMPLEX_TASK:
            # 通用复合任务无确定性模板：先 LLM 分解，失败再回退到意图模板
            plan = self._llm_plan(user_input, intent, context)
            if plan is not None:
                return plan
            return self._template_plan(intent.name, user_input)

        # 简单意图，以及简历优化等自带确定性模板的复杂意图：零 LLM 开销
        return self._template_plan(intent.name, user_input)

    def replan(
        self,
        goal: str,
        trace: List[str],
        failed_step_id: str,
        error: str,
        feedback: str,
    ) -> Optional[TaskPlan]:
        """重新规划剩余步骤；LLM 不可用时返回 None（由编排器降级处理）"""
        if self._llm is None:
            try:
                self._llm = get_shared_llm()
            except Exception as e:
                logger.warning(f"[MasterAgent] 重规划时 LLM 不可用: {e}")
                return None

        prompt = REPLANNER_USER_PROMPT.format(
            goal=goal[:500],
            trace="\n".join(trace[-6:]) or "无",
            failed_step=failed_step_id,
            error=error[:200],
            feedback=feedback[:200] or "无",
            executor_catalog=self.executor_catalog,
        )

        try:
            raw = self._llm.generate(prompt, deepThinking=False, max_new_tokens=256)
            steps = self._parse_steps(raw, require_list=True)
            if not steps:
                return None
            logger.info(f"[MasterAgent] 重规划产出 {len(steps)} 个剩余步骤")
            return TaskPlan(goal=goal, intent="replan", steps=steps, source="replan")
        except Exception as e:
            logger.warning(f"[MasterAgent] 重规划失败: {e}")
            return None

    # ---------- 模板计划 ----------

    def _template_plan(self, intent_name: str, user_input: str) -> TaskPlan:
        """按意图使用确定性模板计划"""
        goal = (user_input or intent_name).strip()[:120] or intent_name

        if intent_name == INTENT_RESUME_OPTIMIZE:
            steps = [
                TaskStep(step_id="step_1", agent="resume_score",
                         task="对简历进行多维度评分", depends_on=[]),
                TaskStep(step_id="step_2", agent="jd_match",
                         task="分析简历与 JD 的匹配度并生成优化建议", depends_on=[]),
                TaskStep(step_id="step_3", agent="resume_polish",
                         task="基于评分与建议润色简历", depends_on=["step_1", "step_2"]),
            ]
        elif intent_name == INTENT_RESUME_SCORE:
            steps = [TaskStep(step_id="step_1", agent="resume_score",
                              task="对简历进行多维度评分", depends_on=[])]
        elif intent_name == INTENT_JD_MATCH:
            steps = [TaskStep(step_id="step_1", agent="jd_match",
                              task="分析简历与 JD 的匹配度", depends_on=[])]
        elif intent_name == INTENT_RESUME_POLISH:
            steps = [TaskStep(step_id="step_1", agent="resume_polish",
                              task="润色简历表述", depends_on=[])]
        elif intent_name == INTENT_RESUME_PARSE:
            steps = [TaskStep(step_id="step_1", agent="resume_parse",
                              task="解析简历文件提取结构化信息", depends_on=[])]
        else:  # 通用对话
            steps = [TaskStep(step_id="step_1", agent="chat",
                              task="回答用户问题", depends_on=[])]

        logger.info(f"[MasterAgent] 模板计划: intent={intent_name}, steps={len(steps)}")
        return TaskPlan(goal=goal, intent=intent_name, steps=steps, source="template")

    # ---------- LLM 计划 ----------

    def _llm_plan(
        self,
        user_input: str,
        intent: IntentResult,
        context: Optional[Dict[str, Any]],
    ) -> Optional[TaskPlan]:
        """用 LLM 分解复杂任务；解析失败返回 None"""
        if self._llm is None:
            try:
                self._llm = get_shared_llm()
            except Exception as e:
                logger.warning(f"[MasterAgent] LLM 不可用，无法分解任务: {e}")
                return None

        context_desc = "无"
        if context:
            keys = [k for k, v in context.items() if v]
            context_desc = ", ".join(keys) if keys else "无"

        system = PLANNER_SYSTEM_PROMPT.format(
            executor_catalog=self.executor_catalog, max_steps=MAX_LLM_STEPS
        )
        user = PLANNER_USER_PROMPT.format(
            user_input=user_input[:800],
            intent=f"{intent.name}（{intent.reason}）",
            context=context_desc,
        )
        prompt = f"{system}\n\n{user}"

        try:
            with start_span("llm-plan", as_type="generation", input=prompt[:500]):
                raw = self._llm.generate(prompt, deepThinking=False, max_new_tokens=320)
                update_current_generation(output=raw[:500])
            steps = self._parse_steps(raw)
            if not steps:
                logger.warning("[MasterAgent] LLM 计划解析为空，回退模板")
                return None

            plan = TaskPlan(
                goal=f"[分解] {user_input[:100]}",
                intent=intent.name,
                steps=steps,
                source="llm",
            )
            logger.info(f"[MasterAgent] LLM 计划: {[(s.step_id, s.agent) for s in steps]}")
            return plan

        except Exception as e:
            logger.warning(f"[MasterAgent] LLM 计划失败: {e}")
            return None

    def _parse_steps(self, raw: str, require_list: bool = False) -> Optional[List[TaskStep]]:
        """从 LLM 输出中解析步骤列表，并校验执行器合法性"""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        raw_steps = data.get("steps") if isinstance(data, dict) else data
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        if require_list and not raw_steps:
            return None

        steps: List[TaskStep] = []
        seen_ids = set()
        for i, item in enumerate(raw_steps[:MAX_LLM_STEPS], start=1):
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent", "")).strip()
            if not self._is_valid_agent(agent):
                logger.warning(f"[MasterAgent] LLM 规划了未知执行器 '{agent}'，丢弃该步骤")
                continue

            step_id = str(item.get("step_id") or f"step_{i}").strip() or f"step_{i}"
            if step_id in seen_ids:
                n = i
                while f"step_{n}" in seen_ids:
                    n += 1
                step_id = f"step_{n}"
            seen_ids.add(step_id)

            deps = item.get("depends_on") or []
            if not isinstance(deps, list):
                deps = []
            deps = [d for d in (str(x).strip() for x in deps) if d in seen_ids]

            steps.append(TaskStep(
                step_id=step_id,
                agent=agent,
                task=str(item.get("task") or "执行任务")[:200],
                depends_on=deps,
            ))

        # 修复悬空依赖（依赖不在本计划中 → 清空）
        plan_ids = {s.step_id for s in steps}
        for s in steps:
            s.depends_on = [d for d in s.depends_on if d in plan_ids and d != s.step_id]

        return steps or None

    def _is_valid_agent(self, agent: str) -> bool:
        if not self.executor_catalog:
            # 无目录时仅放行内置执行器
            return agent in {"resume_score", "jd_match", "resume_polish", "resume_parse", "chat"}
        return f"- {agent}:" in self.executor_catalog
