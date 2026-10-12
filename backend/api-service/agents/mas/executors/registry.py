"""执行器注册表"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from logger import get_logger
from .base import BaseExecutor
from .resume_score import ResumeScoreExecutor
from .jd_match import JDMatchExecutor
from .resume_polish import ResumePolishExecutor
from .resume_parse import ResumeParseExecutor
from .resume_condense import ResumeCondenseExecutor
from .chat import ChatExecutor

logger = get_logger(__name__)


class ExecutorRegistry:
    """执行器注册表：Master 按名称派发任务"""

    def __init__(self, llm: Optional[Any] = None):
        self._executors: Dict[str, BaseExecutor] = {}
        self._llm = llm
        self._register_default()

    def _register_default(self):
        for executor in (
            ResumeScoreExecutor(llm=self._llm),
            JDMatchExecutor(llm=self._llm),
            ResumePolishExecutor(llm=self._llm),
            ResumeCondenseExecutor(llm=self._llm),
            ResumeParseExecutor(),
            ChatExecutor(llm=self._llm),
        ):
            self.register(executor)

    def register(self, executor: BaseExecutor):
        self._executors[executor.name] = executor
        logger.debug(f"[ExecutorRegistry] 注册执行器: {executor.name}")

    def get(self, name: str) -> Optional[BaseExecutor]:
        return self._executors.get(name)

    def has(self, name: str) -> bool:
        return name in self._executors

    def catalog(self) -> str:
        """生成执行器目录文本（供 LLM 规划提示词使用）"""
        return "\n".join(
            f"- {e.name}: {e.description}（必需输入: {', '.join(e.required_inputs) or '无'}）"
            for e in self._executors.values()
        )

    def names(self) -> List[str]:
        return list(self._executors.keys())
