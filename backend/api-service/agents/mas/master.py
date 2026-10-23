"""
Master 编排器：主从多智能体主流程
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional

from logger import get_logger
from services.tracing import start_span, update_observation
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
MAX_RETRIES_PER_STEP = 1  # 单步骤重试次数上限（重试预算统一由编排器控制）


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
            self._max_retries_per_step = mas_cfg.reflection.max_step_retries
            if use_llm_reflection is None:
                use_llm_reflection = mas_cfg.reflection.use_llm
        except Exception as e:
            logger.warning(f"[MasterOrchestrator] 读取 MAS 配置失败，使用默认值: {e}")
            self._max_total_rounds = MAX_TOTAL_ROUNDS
            self._max_replans = MAX_REPLANS
            self._stream_chunk_chars = STREAM_CHUNK_CHARS
            self._max_retries_per_step = MAX_RETRIES_PER_STEP
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
        ) as root_obs:
            for event in self._run_stream_impl(user_input, context, preset_intent):
                if event.get("type") == "complete":
                    data = event.get("data") or {}
                    final_output = {
                        k: data.get(k)
                        for k in ("answer", "overall_score", "steps_success", "steps_total")
                        if data.get(k) is not None
                    }
                    update_observation(root_obs, output=final_output)
                yield event

    def _run_stream_impl(
        self,
        user_input: str,
        context: Dict[str, Any],
        preset_intent: Optional[IntentResult] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流水线实现：意图识别 → Master 规划 → 调度执行 → 反思裁决 → 结果聚合"""
        start_ts = time.time()
        blackboard = Blackboard(message=user_input or "", **context)

        try:
            # ---------- 1. 意图识别 ----------
            with start_span("classify-intent", input=(user_input or "")[:200]) as obs:
                intent = self.intent_recognizer.recognize(
                    user_input, context, preset_intent
                )
                update_observation(obs, output=intent.to_dict())
            yield {"type": "intent", "data": intent.to_dict()}

            # ---------- 2. Master 规划 ----------
            with start_span("plan-tasks", input=(user_input or "")[:200]) as obs:
                plan = self.master_agent.plan(user_input, intent, context)
                update_observation(obs, output={
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

                # 分区：非流式步骤可并行，流式步骤串行
                non_streamable = [s for s in ready if not self._is_streamable(s.agent)]
                streamable = [s for s in ready if self._is_streamable(s.agent)]

                if len(non_streamable) > 1:
                    # ---- 并行批量执行非流式步骤（如评分+匹配）----
                    for step in non_streamable:
                        step.status = STEP_RUNNING
                        yield {"type": "step_start", "data": {
                            "step_id": step.step_id, "agent": step.agent, "task": step.task,
                        }}

                    pool = ThreadPoolExecutor(max_workers=len(non_streamable))
                    futures = {}
                    batch_aborted = False
                    abort_message = ""
                    try:
                        for step in non_streamable:
                            ti = blackboard.snapshot()
                            ti.update(step.params)
                            ti["step_id"] = step.step_id
                            executor = self.executors.get(step.agent)
                            if executor is None:
                                futures[step.step_id] = pool.submit(
                                    lambda s=step: StepResult(
                                        step_id=s.step_id, agent=s.agent,
                                        success=False, error=f"执行器 {s.agent} 不存在",
                                    )
                                )
                            else:
                                futures[step.step_id] = pool.submit(executor.run, ti)

                        for step in non_streamable:
                            try:
                                result = futures[step.step_id].result()
                            except Exception as e:
                                result = StepResult(
                                    step_id=step.step_id, agent=step.agent,
                                    success=False, error=str(e),
                                )

                            verdict, reflection = yield from self._post_execute(
                                step, result, plan, blackboard
                            )
                            aborted, finished, replans, replan_ev = self._handle_verdict(
                                verdict, reflection, step, result, plan, blackboard, replans
                            )
                            if replan_ev:
                                yield replan_ev
                            if aborted:
                                batch_aborted = True
                                abort_message = reflection.thought or "流程被反思器终止"
                                break
                            if finished:
                                finished_early = True
                                break

                        # 流式步骤串行（仅在批次正常结束时执行）
                        if not batch_aborted and not finished_early:
                            for step in streamable:
                                step.status = STEP_RUNNING
                                yield {"type": "step_start", "data": {
                                    "step_id": step.step_id, "agent": step.agent, "task": step.task,
                                }}
                                step_gen = self._execute_step_iter(step, blackboard)
                                try:
                                    while True:
                                        yield next(step_gen)
                                except StopIteration as stop:
                                    result = stop.value

                                verdict, reflection = yield from self._post_execute(
                                    step, result, plan, blackboard
                                )
                                aborted, finished, replans, replan_ev = self._handle_verdict(
                                    verdict, reflection, step, result, plan, blackboard, replans
                                )
                                if replan_ev:
                                    yield replan_ev
                                if aborted:
                                    batch_aborted = True
                                    abort_message = reflection.thought or "流程被反思器终止"
                                    break
                                if finished:
                                    finished_early = True
                                    break
                    finally:
                        # 取消排队中的任务；在途 LLM 调用放后台自行收尾，
                        # 不阻塞 abort 响应（wait=False，线程完成后自动释放）
                        for fut in futures.values():
                            fut.cancel()
                        pool.shutdown(wait=False)

                    if batch_aborted:
                        yield {"type": "error", "message": abort_message}
                        return
                    if finished_early:
                        break

                else:
                    # ---- 原串行逻辑 ----
                    for step in ready:
                        step.status = STEP_RUNNING
                        yield {"type": "step_start", "data": {
                            "step_id": step.step_id, "agent": step.agent, "task": step.task,
                        }}
                        step_gen = self._execute_step_iter(step, blackboard)
                        try:
                            while True:
                                yield next(step_gen)
                        except StopIteration as stop:
                            result = stop.value

                        verdict, reflection = yield from self._post_execute(
                            step, result, plan, blackboard
                        )
                        aborted, finished, replans, replan_ev = self._handle_verdict(
                            verdict, reflection, step, result, plan, blackboard, replans
                        )
                        if replan_ev:
                            yield replan_ev
                        if aborted:
                            yield {"type": "error", "message": reflection.thought or "流程被反思器终止"}
                            return
                        if finished:
                            finished_early = True
                            break

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

    def _is_streamable(self, agent_name: str) -> bool:
        """判断执行器是否支持流式输出"""
        executor = self.executors.get(agent_name)
        return executor is not None and getattr(executor, "streamable", False)

    def _post_execute(
        self,
        step: TaskStep,
        result: StepResult,
        plan: TaskPlan,
        blackboard: Blackboard,
    ) -> Generator[Dict[str, Any], None, tuple]:
        """步骤执行后的通用处理：记录结果、输出 step_result / legacy / reflection 事件。

        返回 (verdict, reflection) 供调用方做循环控制。
        反思器无法裁决（None）时，按编排策略保守合成 retry 裁决。
        """
        blackboard.record(result)
        step.result = result
        step.status = STEP_SUCCESS if result.success else STEP_FAILED
        yield {"type": "step_result", "data": result.to_dict()}
        yield from self._legacy_events(step, result, plan)

        with start_span(
            "reflect",
            input={"step": step.step_id, "agent": step.agent,
                   "success": result.success},
        ) as obs:
            reflection = self.reflector.review(
                step, result, plan.goal, blackboard.trace_lines()
            )
            if reflection is None:
                # 评审链无法裁决 → 控制策略兜底：预算内保守重试
                logger.info(f"[MasterOrchestrator] 步骤 {step.step_id} 评审无法裁决，保守重试")
                reflection = Reflection(
                    step_id=step.step_id, verdict=VERDICT_RETRY,
                    thought="评审器无法裁决，按编排策略保守重试",
                    feedback="请检查输出格式与完整性后重新执行",
                    source="policy",
                )
            update_observation(obs, output=reflection.to_dict())
        yield {"type": "reflection", "data": reflection.to_dict()}
        return reflection.verdict, reflection

    def _handle_verdict(
        self,
        verdict: str,
        reflection: Reflection,
        step: TaskStep,
        result: StepResult,
        plan: TaskPlan,
        blackboard: Blackboard,
        replans: int,
    ) -> tuple:
        """处理反思裁决（控制策略唯一入口），返回 (aborted, finished, updated_replans, replan_event)。

        重试预算统一在此执行：retry 裁决超出预算时升级为重规划（沿用重规划预算）。
        """
        if verdict == VERDICT_CONTINUE:
            return False, False, replans, None

        if verdict == VERDICT_FINISH:
            self._skip_remaining(plan, reason="反思判定目标已达成")
            return False, True, replans, None

        if verdict == VERDICT_ABORT:
            return True, False, replans, None

        if verdict == VERDICT_RETRY and step.retries < self._max_retries_per_step:
            step.retries += 1
            step.status = STEP_PENDING
            step.params["_feedback"] = reflection.feedback
            logger.info(
                f"[MasterOrchestrator] 步骤 {step.step_id} 重试 "
                f"({step.retries}/{self._max_retries_per_step}): {reflection.feedback[:80]}"
            )
            return False, False, replans, None

        if verdict in (VERDICT_RETRY, VERDICT_REPLAN):
            # retry 预算耗尽 / 显式 replan：走重规划
            return self._do_replan(reflection, step, result, plan, blackboard, replans)

        # 未知 verdict：默认继续
        return False, False, replans, None

    def _do_replan(
        self,
        reflection: Reflection,
        step: TaskStep,
        result: StepResult,
        plan: TaskPlan,
        blackboard: Blackboard,
        replans: int,
    ) -> tuple:
        """执行重规划；预算用尽则跳过失败步骤"""
        if replans >= self._max_replans:
            step.status = STEP_SKIPPED
            logger.warning("[MasterOrchestrator] 重规划次数用尽，跳过失败步骤")
            return False, False, replans, None
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
            return False, False, replans, {"type": "plan", "data": plan.to_dict()}
        step.status = STEP_SKIPPED
        return False, False, replans, None

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
        ) as obs:
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
            update_observation(
                obs,
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
            elif ctype == "thought":
                # ReAct 步内思考事件：实时透传给前端
                yield chunk
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
                "dimension_analysis": result.data.get("dimension_analysis") or {},
            }}
        elif step.agent == "jd_match":
            yield {"type": "suggestions", "data": {
                "suggestions": result.data.get("suggestions"),
                "match_analysis": result.data.get("match_result"),
            }}
        elif step.agent == "resume_condense":
            yield {"type": "condensed", "data": {
                "condensed_resume": result.data.get("condensed_resume"),
                "original_length": result.data.get("original_length"),
                "condensed_length": result.data.get("condensed_length"),
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
            "dimension_analysis": "dimension_analysis",
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
