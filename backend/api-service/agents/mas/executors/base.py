"""执行器基类与共享工具"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Generator, List, Optional

from logger import get_logger
from services.tracing import start_span, update_current_generation
from agents.llm import get_shared_llm
from ..prompts import REACT_DECIDE_SYSTEM_PROMPT, REACT_DECIDE_USER_PROMPT
from ..state import StepResult

logger = get_logger(__name__)


def _resolve_llm(injected: Optional[Any], task_input: Dict[str, Any]) -> Any:
    """解析执行器使用的 LLM：构造时注入的优先，否则按请求参数切换 Provider"""
    if injected is not None:
        return injected
    return get_shared_llm(
        provider=task_input.get("provider"),
        model=task_input.get("model"),
    )


class ReActLoop:
    """步内 ReAct 循环：Thought → Action → Observation

    与 Reflector（步后评审）无关——ReAct 是执行器步内的推理-行动循环。

    设计约束（本地小模型友好）：
    - 决策轮为非流式严格 JSON（thought/action/query），解析失败按 answer 兜底
    - 工具轮数受 max_tool_rounds 限制，预算耗尽强制进入最终回答
    - 最终回答由执行器自行流式输出，本类只负责"要不要查、查什么"
    """

    def __init__(
        self,
        llm: Any,
        tools: Dict[str, Callable[[str], str]],
        max_tool_rounds: int = 1,
        max_new_tokens: int = 256,
    ):
        self.llm = llm
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.max_new_tokens = max_new_tokens

    def decide(self, message: str, observation: str) -> tuple:
        """一轮决策：返回 (thought, action, query)；解析失败/未知行动按 answer 兜底"""
        prompt = (
            f"{REACT_DECIDE_SYSTEM_PROMPT}\n\n"
            + REACT_DECIDE_USER_PROMPT.format(
                message=message[:500],
                observation=observation[:800] or "无",
            )
        )
        try:
            with start_span("llm-react-decide", as_type="generation", input=prompt[:500]):
                raw = self.llm.generate(
                    prompt, deepThinking=False, max_new_tokens=self.max_new_tokens
                )
                update_current_generation(output=(raw or "")[:300])
            thought, action, query = self._parse_decision(raw or "")
        except Exception as e:
            logger.warning(f"[ReActLoop] 决策轮失败，直接回答: {e}")
            return "", "answer", ""

        if action not in self.tools:
            action = "answer"
        return thought, action, query

    def run_tool(self, action: str, query: str) -> str:
        """执行工具，返回 Observation 文本；工具失败返回空串"""
        tool = self.tools.get(action)
        if tool is None:
            return ""
        try:
            return (tool(query) or "").strip()
        except Exception as e:
            logger.warning(f"[ReActLoop] 工具 {action} 执行失败: {e}")
            return ""

    @staticmethod
    def _parse_decision(raw: str) -> tuple:
        from services.json_utils import extract_json_object
        json_str = extract_json_object(raw)
        if not json_str:
            return "", "answer", ""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return "", "answer", ""
        thought = str(data.get("thought", ""))[:200]
        action = str(data.get("action", "answer")).strip().lower()
        query = str(data.get("query", "")).strip()
        return thought, action, query


class BaseExecutor:
    """执行器基类"""

    name: str = "base"
    description: str = ""
    # 必需输入键（缺失时执行器快速失败，供反思器判定重试）
    required_inputs: List[str] = []
    # 是否支持流式输出（Master 据此选择 run_stream / run）
    streamable: bool = False

    def validate(self, task_input: Dict[str, Any]) -> Optional[str]:
        """校验输入，返回错误信息（None 表示通过）"""
        missing = [k for k in self.required_inputs if not task_input.get(k)]
        if missing:
            return f"缺少必需输入: {', '.join(missing)}"
        return None

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        """同步执行（子类必须实现）"""
        raise NotImplementedError

    def run_stream(self, task_input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """流式执行（可选实现，默认降级为同步 + 单块输出）"""
        result = self.run(task_input)
        if result.success and result.data.get("_stream_text"):
            text = result.data.pop("_stream_text")
            yield {"type": "token", "content": text}
        elif not result.success:
            yield {"type": "error", "content": result.error}

    def _new_result(self, step_id: str = "") -> StepResult:
        return StepResult(step_id=step_id, agent=self.name)
