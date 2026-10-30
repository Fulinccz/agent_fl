"""面试攻略执行器"""
from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Generator, Optional

from logger import get_logger
from .base import BaseExecutor, _resolve_llm
from ..state import StepResult

logger = get_logger(__name__)


class InterviewPrepExecutor(BaseExecutor):
    """面试攻略执行器：基于简历与目标岗位生成整体攻略与高频问题"""

    name = "interview_prep"
    description = "基于简历与目标岗位生成面试整体攻略与高频问题"
    required_inputs = ["resume"]
    # 整体攻略与三维度并发，完成后按维度顺序推送增量（首个增量约 1 分钟）
    streamable = True

    # 增量之间的心跳间隔：维度生成期间零字节会让中间代理误判连接空闲，
    # 心跳保证 SSE 连接始终有数据流动
    HEARTBEAT_SECONDS = 10

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            agent = self._build_agent(task_input)
            state = self._new_state(task_input)
            agent.run(state)

            # 兜底错误不写入 result.error（Reflector 会判 RETRY 造成无效重试）；
            # 该 agent 有规则题库兜底，总有产出，错误信息随 data 保留供排查
            if state.get("error"):
                result.data = {"interview_fallback_error": state["error"]}

            result.success = True
            result.data.update({"interview_guide": state.get("interview_guide")})
            guide = state.get("interview_guide") or {}
            count = len(guide.get("questions") or [])
            result.summary = f"生成 {count} 个高频面试问题"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result

    def run_stream(
        self, task_input: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行：题卡增量 + 等待期心跳

        agent 的出题调用是阻塞的，放到后台线程跑、用队列回传增量；
        三维度并发完成后按顺序推送，前端一次性拿到完整题库。
        """
        error = self.validate(task_input)
        if error:
            yield {"type": "error", "content": error}
            return

        chunks: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        state = self._new_state(task_input)
        # 消费者中途关闭生成器（客户端断开）时置位，agent 据此跳过剩余维度
        stop = threading.Event()

        def _emit(guide: Dict[str, Any]) -> None:
            chunks.put({
                "type": "interview",
                "data": {"interview_guide": guide, "partial": True},
            })

        def _worker() -> None:
            try:
                agent = self._build_agent(
                    task_input, on_progress=_emit, should_stop=stop.is_set
                )
                agent.run(state)
            except Exception as e:
                logger.error(f"[{self.name}] 流式执行异常: {e}")
            finally:
                chunks.put(None)

        threading.Thread(target=_worker, name="interview-prep", daemon=True).start()

        try:
            while True:
                try:
                    chunk = chunks.get(timeout=self.HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield {"type": "heartbeat"}
                    continue
                if chunk is None:
                    break
                yield chunk
        finally:
            stop.set()

    def _build_agent(self, task_input: Dict[str, Any], **kwargs: Any) -> Any:
        """构造 interview agent：LLM 解析失败不抛错，降级为规则题库"""
        from agents.skills.resume_agents.interview_agent import InterviewPrepAgent

        try:
            llm = _resolve_llm(self._llm, task_input)
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 不可用，改用规则题库: {e}")
            llm = None
        return InterviewPrepAgent(llm=llm, **kwargs)

    @staticmethod
    def _new_state(task_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "resume": task_input["resume"],
            "jd": task_input.get("jd"),
            # 页面入口的场景指令（面试页：面试官视角出题）
            "directive": task_input.get("directive"),
            "interview_guide": None,
            "error": None,
            "current_step": "started",
        }
