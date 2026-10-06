"""Master 编排器：主从多智能体主流程

完整流水线：
    用户输入 → 意图识别 → Master 规划 → 步骤调度执行（Slave）
             → ReAct 反思（继续/重试/重规划/终止）→ 结果聚合 → 事件流输出

事件协议（每行一个 JSON，与现有 SSE 契约兼容）：
    intent / plan / step_start / step_result / reflection   —— 新增诊断事件（前端自动忽略）
    score / suggestions / polished / complete / error        —— 简历优化旧契约（保持不变）
    token / complete / error                                 —— 对话旧契约（保持不变）
"""
from __future__ import annotations

import time
from typing import Any, Dict, Generator, List, Optional

from logger import get_logger
from services.tracing import start_span, update_current_span
from .intent import IntentRecognizer
from .planner import MasterAgent
from .reflector import Reflector
from .executors import ExecutorRegistry, BaseExecutor
from .state import (
    Blackboard,
    IntentResult,
    Reflection,
    StepResult,
    TaskPlan,
    TaskStep,
    STEP_PENDING,
    STEP_RUNNING,
    STEP_SUCCESS,
    STEP_FAILED,
    STEP_SKIPPED,
    VERDICT_CONTINUE,
    VERDICT_RETRY,
    VERDICT_REPLAN,
    VERDICT_FINISH,
    VERDICT_ABORT,
)

logger = get_logger(__name__)

# 流式输出节流/轮数默认值（可被 config.yaml [mas] 小节覆盖）
STREAM_CHUNK_CHARS = 50
MAX_TOTAL_ROUNDS = 12   # 调度总轮数上限（防死循环）
MAX_REPLANS = 1         # 重规划次数上限


class MasterOrchestrator:
    """主从多智能体编排器"""

    def __init__(
        self,
        llm: Optional[Any] = None,
        enable_vector_intent: bool = True,
        use_llm_reflection: Optional[bool] = None,
    ):
        # 编排参数：config.yaml [mas] 小节（代码默认值兜底）
        try:
            from services.app_config import load_app_config
            mas_cfg = load_app_config().mas
            self._max_total_rounds = mas_cfg.max_total_rounds
            self._max_replans = mas_cfg.max_replans
            self._stream_chunk_chars = mas_cfg.stream_chunk_chars
            if use_llm_reflection is None:
                use_llm_reflection = mas_cfg.reflection.use_llm
        except Exception as e:
            logger.warning(f"[MasterOrchestrator] 读取 MAS 配置失败，使用默认值: {e}")
            self._max_total_rounds = MAX_TOTAL_ROUNDS
            self._max_replans = MAX_REPLANS
            self._stream_chunk_chars = STREAM_CHUNK_CHARS
            if use_llm_reflection is None:
                use_llm_reflection = True

        self.executors = ExecutorRegistry(llm=llm)
        self.intent_recognizer = IntentRecognizer(
            llm=llm, enable_vector=enable_vector_intent
        )
        self.master_agent = MasterAgent(
            llm=llm, executor_catalog=self.executors.catalog()
        )
        self.reflector = Reflector(llm=llm, use_llm_reflection=use_llm_reflection)
        logger.info("[MasterOrchestrator] 初始化完成，执行器: %s", self.executors.names())

    # ============ 非流式接口 ============

    def run(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        preset_intent: Optional[IntentResult] = None,
    ) -> Dict[str, Any]:
        """同步执行完整流水线，返回聚合结果"""
        answer_parts: List[str] = []
        final_state: Dict[str, Any] = {}

        for event in self.run_stream(user_input, context, preset_intent):
            etype = event.get("type")
            if etype == "token":
                answer_parts.append(event.get("content", ""))
            elif etype == "polished" and not event.get("data", {}).get("partial"):
                final_state["optimized_resume"] = event["data"].get("optimized_resume")
            elif etype == "score":
                final_state["score"] = event.get("data")
            elif etype == "suggestions":
                final_state["suggestions"] = event.get("data")
            elif etype == "error":
                final_state["error"] = event.get("message")
            elif etype == "complete":
                final_state.update(event.get("data") or {})

        return {
            "success": "error" not in final_state,
            "answer": "".join(answer_parts) or final_state.get("optimized_resume", ""),
            "intent": final_state.get("intent"),
            **final_state,
        }

    # ============ 流式主流程 ============

    def run_stream(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        preset_intent: Optional[IntentResult] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行完整流水线，逐个 yield 事件（外层包 Langfuse 跟踪）

        根 observation 承载 trace 级属性：session_id（会话分组）、tags、
        最终 output（tracing 表与评估器读取的 trace 输入/输出）。
        """
        context = context or {}
        final_output: Dict[str, Any] = {}
        with start_span(
            "mas-pipeline",
            as_type="agent",
            input=(user_input or "")[:500],
            metadata={
                "provider": context.get("provider", "local"),
                "model": context.get("model", ""),
            },
            session_id=context.get("session_id"),
            tags=["mas"],
        ):
            for event in self._run_stream_impl(user_input, context, preset_intent):
                if event.get("type") == "complete":
                    data = event.get("data") or {}
                    final_output = {
                        k: data.get(k)
                        for k in ("answer", "overall_score", "steps_success", "steps_total")
                        if data.get(k) is not None
                    }
                    update_current_span(output=final_output)
                yield event

    def _run_stream_impl(
        self,
        user_input: str,
        context: Dict[str, Any],
        preset_intent: Optional[IntentResult] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流水线实现：意图识别 → Master 规划 → 调度执行 + ReAct 反思 → 结果聚合"""
        start_ts = time.time()
        blackboard = Blackboard(message=user_input or "", **context)

        try:
            # ---------- 1. 意图识别 ----------
            with start_span("classify-intent", input=(user_input or "")[:200]):
                intent = self.intent_recognizer.recognize(
                    user_input, context, preset_intent
                )
                update_current_span(output=intent.to_dict())
            yield {"type": "intent", "data": intent.to_dict()}

            # ---------- 2. Master 规划 ----------
            with start_span("plan-tasks", input=(user_input or "")[:200]):
                plan = self.master_agent.plan(user_input, intent, context)
                update_current_span(output={
                    "source": plan.source,
                    "steps_count": len(plan.steps),
                    "intent": plan.intent,
                })
            yield {"type": "plan", "data": plan.to_dict()}
            logger.info(f"[MasterOrchestrator] 计划: {plan.to_dict()}")

            # ---------- 3. 调度执行 + ReAct 反思 ----------
            replans = 0
            finished_early = False

            for round_no in range(self._max_total_rounds):
                ready = self._ready_steps(plan)
                if not ready:
                    break

                for step in ready:
                    step.status = STEP_RUNNING
                    yield {"type": "step_start", "data": {
                        "step_id": step.step_id, "agent": step.agent, "task": step.task,
                    }}

                    step_gen = self._execute_step_iter(step, blackboard)
                    try:
                        while True:
                            event = next(step_gen)
                            # 流式事件随生成实时透传（对话 token / 润色 polished）
                            yield event
                    except StopIteration as stop:
                        result = stop.value
                    blackboard.record(result)

                    step.result = result
                    step.status = STEP_SUCCESS if result.success else STEP_FAILED
                    yield {"type": "step_result", "data": result.to_dict()}

                    # 旧契约兼容：简历流程阶段事件
                    yield from self._legacy_events(step, result, plan)

                    # ---------- ReAct 反思 ----------
                    with start_span(
                        "reflect",
                        input={"step": step.step_id, "agent": step.agent,
                               "success": result.success},
                    ):
                        reflection = self.reflector.review(
                            step, result, plan.goal, blackboard.trace_lines()
                        )
                    update_current_span(output=reflection.to_dict())
                    yield {"type": "reflection", "data": reflection.to_dict()}

                    verdict = reflection.verdict
                    if verdict == VERDICT_CONTINUE:
                        continue

                    elif verdict == VERDICT_FINISH:
                        self._skip_remaining(plan, reason="反思判定目标已达成")
                        finished_early = True
                        break

                    elif verdict == VERDICT_RETRY:
                        step.retries += 1
                        step.status = STEP_PENDING
                        step.params["_feedback"] = reflection.feedback
                        logger.info(
                            f"[MasterOrchestrator] 步骤 {step.step_id} 重试 "
                            f"({step.retries}): {reflection.feedback[:80]}"
                        )

                    elif verdict == VERDICT_REPLAN:
                        if replans >= self._max_replans:
                            step.status = STEP_SKIPPED
                            logger.warning("[MasterOrchestrator] 重规划次数用尽，跳过失败步骤")
                            continue
                        replans += 1
                        new_plan = self.master_agent.replan(
                            goal=plan.goal,
                            trace=blackboard.trace_lines(),
                            failed_step_id=step.step_id,
                            error=result.error or result.summary,
                            feedback=reflection.feedback,
                        )
                        new_steps = new_plan.steps if new_plan is not None else None
                        if new_steps and self._apply_replan(plan, new_steps, failed=step):
                            yield {"type": "plan", "data": plan.to_dict()}
                        else:
                            # 重规划失败 → 跳过失败步骤继续
                            step.status = STEP_SKIPPED

                    elif verdict == VERDICT_ABORT:
                        yield {"type": "error", "message": reflection.thought or "流程被反思器终止"}
                        return

                if finished_early:
                    break
            else:
                logger.warning("[MasterOrchestrator] 达到最大调度轮数，强制结束")

            # ---------- 4. 结果聚合 ----------
            yield from self._finalize(plan, blackboard, intent, start_ts)

        except Exception as e:
            logger.error(f"[MasterOrchestrator] 流程异常: {e}", exc_info=True)
            yield {"type": "error", "message": f"编排流程异常: {e}"}

    # ============ 内部方法 ============

    @staticmethod
    def _ready_steps(plan: TaskPlan) -> List[TaskStep]:
        """依赖已满足的待执行步骤（保持计划顺序）"""
        done = {s.step_id for s in plan.steps if s.status == STEP_SUCCESS}
        skipped = {s.step_id for s in plan.steps if s.status == STEP_SKIPPED}
        ready = []
        for step in plan.steps:
            if step.status != STEP_PENDING:
                continue
            # 依赖步骤被跳过也视为满足（避免死锁）
            if all(d in done or d in skipped for d in step.depends_on):
                ready.append(step)
        return ready

    def _execute_step_iter(
        self, step: TaskStep, blackboard: Blackboard
    ) -> Generator[Dict[str, Any], None, StepResult]:
        """执行单个步骤：实时 yield 流式事件（token/polished），return StepResult

        调用方需迭代至 StopIteration 取回结果（stop.value），事件随生成实时透传。
        """
        executor = self.executors.get(step.agent)
        if executor is None:
            return StepResult(
                step_id=step.step_id, agent=step.agent, success=False,
                error=f"执行器 {step.agent} 不存在",
            )

        task_input = blackboard.snapshot()
        task_input.update(step.params)
        task_input["step_id"] = step.step_id
        feedback = step.params.get("_feedback")
        if feedback:
            task_input["_feedback"] = feedback

        with start_span(
            f"step:{step.agent}",
            input={"step_id": step.step_id, "task": (step.task or "")[:200]},
        ):
            try:
                if getattr(executor, "streamable", False):
                    result = yield from self._iter_streaming(executor, step, task_input)
                else:
                    result = executor.run(task_input)
            except Exception as e:
                logger.error(f"[MasterOrchestrator] 步骤 {step.step_id} 执行异常: {e}")
                result = StepResult(
                    step_id=step.step_id, agent=step.agent, success=False, error=str(e)
                )
            update_current_span(
                output={"success": result.success, "summary": result.summary,
                        "error": result.error}
            )
        return result

    def _iter_streaming(
        self,
        executor: BaseExecutor,
        step: TaskStep,
        task_input: Dict[str, Any],
    ) -> Generator[Dict[str, Any], None, StepResult]:
        """流式执行器：token 随生成实时 yield，同时聚合出 StepResult"""
        accumulated = ""
        last_yielded = 0
        has_error = ""

        for chunk in executor.run_stream(task_input):
            ctype = chunk.get("type")
            if ctype == "token":
                content = chunk.get("content", "")
                accumulated += content
                if executor.name == "chat":
                    # 对话：逐 token 实时透传
                    yield {"type": "token", "content": content}
                elif executor.name == "resume_polish":
                    # 润色：按累积长度节流，与旧 polished(partial) 契约一致
                    if len(accumulated) - last_yielded >= self._stream_chunk_chars:
                        yield {
                            "type": "polished",
                            "data": {"optimized_resume": accumulated, "partial": True},
                        }
                        last_yielded = len(accumulated)
            elif ctype == "error":
                has_error = chunk.get("content", "生成错误")

        if has_error and not accumulated:
            return StepResult(
                step_id=step.step_id, agent=step.agent, success=False, error=has_error
            )

        if executor.name == "resume_polish":
            # 最终清理（复用 polish agent 的清理逻辑）
            final_text = self._finalize_polish(executor, accumulated, task_input)
            yield {
                "type": "polished",
                "data": {"optimized_resume": final_text, "partial": False},
            }
            return StepResult(
                step_id=step.step_id, agent=step.agent,
                success=bool(final_text),
                data={"optimized_resume": final_text},
                summary=f"润色完成，输出 {len(final_text)} 字",
                error="" if final_text else "润色输出为空",
            )

        # chat
        answer = accumulated.strip()
        return StepResult(
            step_id=step.step_id, agent=step.agent,
            success=bool(answer),
            data={"answer": answer},
            summary=f"生成回复 {len(answer)} 字",
            error="" if answer else "对话生成为空",
        )

    @staticmethod
    def _finalize_polish(executor: BaseExecutor, raw_text: str, task_input: Dict[str, Any]) -> str:
        """润色结果最终清理"""
        finalize = getattr(executor, "finalize", None)
        if finalize is not None:
            return finalize(raw_text, task_input.get("resume", ""), task_input)
        return raw_text.strip()

    def _legacy_events(
        self, step: TaskStep, result: StepResult, plan: TaskPlan
    ) -> Generator[Dict[str, Any], None, None]:
        """保持 /resume/optimize/stream 旧事件契约"""
        if not result.success:
            return
        if step.agent == "resume_score":
            yield {"type": "score", "data": {
                "overall_score": result.data.get("overall_score"),
                "scores": result.data.get("score_result"),
            }}
        elif step.agent == "jd_match":
            yield {"type": "suggestions", "data": {
                "suggestions": result.data.get("suggestions"),
                "match_analysis": result.data.get("match_result"),
            }}

    def _apply_replan(
        self, plan: TaskPlan, new_steps: List[TaskStep], failed: TaskStep
    ) -> bool:
        """将重规划的步骤替换掉计划中剩余的待执行步骤"""
        if not new_steps:
            return False
        remaining = [s for s in plan.steps if s.status == STEP_PENDING]
        for old in remaining:
            old.status = STEP_SKIPPED

        # 新步骤 id 可能与历史（失败/已跳过）步骤撞车 → 重命名并同步依赖，
        # 避免 _ready_steps 依赖死锁与 Blackboard 结果覆盖
        existing_ids = {s.step_id for s in plan.steps}
        rename: Dict[str, str] = {}
        for new_step in new_steps:
            original_id = new_step.step_id
            if original_id in existing_ids:
                n = 1
                while f"{original_id}_r{n}" in existing_ids:
                    n += 1
                new_step.step_id = f"{original_id}_r{n}"
                rename[original_id] = new_step.step_id
            existing_ids.add(new_step.step_id)
        for new_step in new_steps:
            new_step.depends_on = [rename.get(d, d) for d in new_step.depends_on]

        plan.steps.extend(new_steps)
        plan.source = "replan"
        logger.info(
            f"[MasterOrchestrator] 重规划: 替换 {len(remaining)} 个待执行步骤, "
            f"新增 {len(new_steps)} 个"
        )
        return True

    @staticmethod
    def _skip_remaining(plan: TaskPlan, reason: str = ""):
        for step in plan.steps:
            if step.status == STEP_PENDING:
                step.status = STEP_SKIPPED
        if reason:
            logger.info(f"[MasterOrchestrator] {reason}")

    def _finalize(
        self,
        plan: TaskPlan,
        blackboard: Blackboard,
        intent: IntentResult,
        start_ts: float,
    ) -> Generator[Dict[str, Any], None, None]:
        """聚合最终结果，产出 complete 事件"""
        snapshot = blackboard.snapshot()
        elapsed = round(time.time() - start_ts, 2)

        complete_data: Dict[str, Any] = {
            "intent": intent.to_dict(),
            "plan": plan.to_dict(),
            "elapsed_seconds": elapsed,
            "steps_total": len(plan.steps),
            "steps_success": sum(1 for s in plan.steps if s.status == STEP_SUCCESS),
        }

        # 简历优化产物（旧契约 complete 结构：黑板字段名 → 旧事件字段名）
        artifact_map = {
            "overall_score": "overall_score",
            "scores": "score_result",
            "suggestions": "suggestions",
            "match_analysis": "match_result",
            "optimized_resume": "optimized_resume",
        }
        for out_key, snap_key in artifact_map.items():
            if snapshot.get(snap_key) is not None:
                complete_data[out_key] = snapshot[snap_key]

        # 对话产物
        chat_results = [r for r in blackboard.results.values()
                        if r.agent == "chat" and r.success]
        if chat_results:
            complete_data["answer"] = chat_results[-1].data.get("answer", "")

        yield {"type": "complete", "data": complete_data}
        logger.info(f"[MasterOrchestrator] 流程完成，耗时 {elapsed}s")
