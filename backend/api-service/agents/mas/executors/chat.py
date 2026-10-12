"""通用对话执行器（流式）"""
from __future__ import annotations

import re
from typing import Any, Dict, Generator, Optional

from logger import get_logger
from services.tracing import start_span, update_current_generation
from ..prompts import CHAT_SYSTEM_PROMPT, CHAT_WITH_RAG_PROMPT
from .base import BaseExecutor, _resolve_llm
from ..state import StepResult

logger = get_logger(__name__)


class ChatExecutor(BaseExecutor):
    """通用对话执行器（流式）"""

    name = "chat"
    description = "结合 RAG 上下文进行通用对话答疑"
    required_inputs = ["message"]
    streamable = True

    def __init__(self, llm: Optional[Any] = None, max_new_tokens: int = 512):
        self._llm = llm
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, task_input: Dict[str, Any]) -> str:
        message = task_input["message"]
        rag_context = task_input.get("rag_context")

        if rag_context:
            body = CHAT_WITH_RAG_PROMPT.format(rag_context=rag_context, message=message)
        else:
            body = message

        return f"{CHAT_SYSTEM_PROMPT}\n\n{body}\n\n助手："

    @staticmethod
    def _clean(text: str) -> str:
        """清理思考标签与角色标记"""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^assistant[:：]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\nassistant[:：]\s*", "\n", text, flags=re.IGNORECASE)
        return text.strip()

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            llm = _resolve_llm(self._llm, task_input)
            prompt = self._build_prompt(task_input)
            with start_span("llm-chat", as_type="generation", input=prompt[:500]):
                text = llm.generate(prompt, deepThinking=False, max_new_tokens=self.max_new_tokens)
                update_current_generation(output=(text or "")[:500])
            text = self._clean(text or "")

            if not text:
                result.error = "对话生成为空"
                return result

            result.success = True
            result.data = {"answer": text, "_stream_text": text}
            result.summary = f"生成回复 {len(text)} 字"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result

    def run_stream(self, task_input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """流式对话：yield {"type": "token", "content": ...}"""
        error = self.validate(task_input)
        if error:
            yield {"type": "error", "content": error}
            return

        try:
            llm = _resolve_llm(self._llm, task_input)
            prompt = self._build_prompt(task_input)
            with start_span("llm-chat-stream", as_type="generation", input=prompt[:500]):
                accumulated = ""
                for chunk in llm.generate_stream(
                    prompt, deepThinking=False, max_new_tokens=self.max_new_tokens
                ):
                    if chunk.get("type") == "token":
                        content = chunk.get("content", "")
                        accumulated += content
                        yield {"type": "token", "content": content}
                    elif chunk.get("type") == "error":
                        yield {"type": "error", "content": chunk.get("content", "生成错误")}
                        return
                update_current_generation(output=accumulated[:500])
        except Exception as e:
            logger.error(f"[{self.name}] 流式执行失败: {e}")
            yield {"type": "error", "content": str(e)}
