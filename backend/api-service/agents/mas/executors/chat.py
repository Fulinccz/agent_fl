"""通用对话执行器（流式）"""
from __future__ import annotations

import re
from typing import Any, Dict, Generator, Optional

from logger import get_logger
from services.tracing import start_span, update_observation
from ..prompts import CHAT_SYSTEM_PROMPT, CHAT_WITH_RAG_PROMPT
from .base import BaseExecutor, ReActLoop, _resolve_llm
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

    # ---------- ReAct 步内回路 ----------

    def _build_react_loop(
        self, task_input: Dict[str, Any], llm: Any
    ) -> Optional[ReActLoop]:
        """按需构建 ReAct 循环；返回 None 表示走直接回答快速路径。

        触发条件（全部满足）：config/请求启用 react，且无预检索上下文
        （已有 rag_context 时直接回答，避免无谓的首字延迟）。
        """
        enabled = task_input.get("use_react")
        max_rounds = 1
        if enabled is None:
            try:
                from services.app_config import load_app_config
                react_cfg = load_app_config().mas.react
                enabled = react_cfg.enabled
                max_rounds = react_cfg.max_tool_rounds
            except Exception as e:
                logger.debug(f"[{self.name}] 读取 react 配置失败，默认关闭: {e}")
                enabled = False
        if not enabled or task_input.get("rag_context"):
            return None
        return ReActLoop(
            llm=llm,
            tools={"search_knowledge": self._tool_search_knowledge},
            max_tool_rounds=max_rounds,
        )

    @staticmethod
    def _tool_search_knowledge(query: str) -> str:
        """ReAct 工具：查询混合检索器（向量 + 图谱），返回拼接的资料文本"""
        try:
            from api.routes.deps import get_hybrid_retriever
            retriever = get_hybrid_retriever()
            results = retriever.retrieve(query, top_k=3) or []
        except Exception as e:
            logger.warning(f"[chat] ReAct 知识库检索失败: {e}")
            return ""
        pieces = []
        for item in results:
            content = (item.get("content") or "").strip()
            if content:
                pieces.append(content)
        return "\n".join(pieces)

    def _build_prompt(self, task_input: Dict[str, Any]) -> str:
        return self._build_prompt_with_context(
            task_input["message"], task_input.get("rag_context")
        )

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
            with start_span("llm-chat", as_type="generation", input=prompt[:500]) as obs:
                text = llm.generate(prompt, deepThinking=False, max_new_tokens=self.max_new_tokens)
                update_observation(obs, output=(text or "")[:500])
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
        """流式对话：yield {"type": "thought"|"token", "content": ...}

        可选 ReAct 前置回路：无预检索上下文时先决策是否查知识库，
        thought 事件实时透传，最终回答仍逐 token 流式。
        """
        error = self.validate(task_input)
        if error:
            yield {"type": "error", "content": error}
            return

        try:
            llm = _resolve_llm(self._llm, task_input)
            message = task_input["message"]
            rag_context = task_input.get("rag_context")

            # ---------- ReAct 前置回路（可选） ----------
            react = self._build_react_loop(task_input, llm)
            if react is not None:
                observation = ""
                for round_no in range(react.max_tool_rounds):
                    thought, action, query = react.decide(message, observation)
                    if thought:
                        yield {"type": "thought", "content": thought}
                    if action != "search_knowledge":
                        break
                    observation = react.run_tool(action, query)
                if observation:
                    rag_context = observation

            prompt = self._build_prompt_with_context(message, rag_context)
            with start_span("llm-chat-stream", as_type="generation", input=prompt[:500]) as obs:
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
                update_observation(obs, output=accumulated[:500])
        except Exception as e:
            logger.error(f"[{self.name}] 流式执行失败: {e}")
            yield {"type": "error", "content": str(e)}

    @staticmethod
    def _build_prompt_with_context(message: str, rag_context: Optional[str]) -> str:
        """组装最终回答提示词（复用既有对话提示词）"""
        if rag_context:
            body = CHAT_WITH_RAG_PROMPT.format(rag_context=rag_context, message=message)
        else:
            body = message
        return f"{CHAT_SYSTEM_PROMPT}\n\n{body}\n\n助手："
