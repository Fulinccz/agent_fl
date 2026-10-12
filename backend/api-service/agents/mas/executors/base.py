"""执行器基类与共享工具"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from logger import get_logger
from agents.llm import get_shared_llm
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
