"""执行器（Slave Agent）定义

每个执行器封装一项原子能力，接受黑板快照 + 步骤参数作为输入，
返回统一的 StepResult。Master 只依赖执行器名称与输入输出契约。

内置执行器：
- resume_score   简历评分        （复用 ResumeScoreAgent）
- jd_match       JD 匹配分析     （复用 JDMatchAgent）
- resume_polish  简历润色        （复用 ResumePolishAgent，支持流式）
- resume_parse   简历文件解析    （复用 rag.document_processor）
- chat           通用对话        （直接调用共享 LLM，支持流式）
"""
from __future__ import annotations

import re
from typing import Any, Dict, Generator, List, Optional

from logger import get_logger
from services.tracing import start_span, update_current_generation
from agents.llm import get_shared_llm
from .state import StepResult
from .prompts import CHAT_SYSTEM_PROMPT, CHAT_WITH_RAG_PROMPT

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


class ResumeScoreExecutor(BaseExecutor):
    """简历评分执行器"""

    name = "resume_score"
    description = "对简历进行多维度评分（完整性/专业度/量化程度/匹配度）"
    required_inputs = ["resume"]

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.score_agent import ResumeScoreAgent

            agent = ResumeScoreAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "score_result": None,
                "overall_score": None,
                "suggestions": None,
                "match_result": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            agent.run(state)

            if state.get("error"):
                result.error = state["error"]
                # 评分 agent 失败时会写入兜底分数，仍视为部分产出
                result.data = {
                    "score_result": state.get("score_result"),
                    "overall_score": state.get("overall_score"),
                }
                result.summary = "评分执行出错，已使用兜底分数"
                return result

            result.success = True
            result.data = {
                "score_result": state.get("score_result"),
                "overall_score": state.get("overall_score"),
            }
            overall = state.get("overall_score") or {}
            result.summary = f"评分完成: {overall.get('score', '?')}分 ({overall.get('rating', '')})"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result


class JDMatchExecutor(BaseExecutor):
    """JD 匹配分析执行器"""

    name = "jd_match"
    description = "分析简历与目标岗位 JD 的匹配度并生成优化建议"
    required_inputs = ["resume"]  # jd 可选，缺失时产出默认建议

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.match_agent import JDMatchAgent

            agent = JDMatchAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "score_result": task_input.get("score_result") or {
                    "completeness": 70, "professionalism": 70,
                    "quantification": 60, "matching": 70,
                },
                "match_result": None,
                "suggestions": None,
                "overall_score": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            agent.run(state)

            # 兜底错误不写入 result.error（Reflector 会判 RETRY 造成无效重试）；
            # 该 agent 设计上总有产出，错误信息随 data 保留供排查
            if state.get("error"):
                result.data = {"match_fallback_error": state["error"]}

            result.success = True  # 匹配 agent 有完整兜底逻辑，总有产出
            result.data.update({
                "match_result": state.get("match_result"),
                "suggestions": state.get("suggestions"),
            })
            match = state.get("match_result") or {}
            result.summary = f"匹配分析完成: 匹配度{match.get('match_score', '?')}分"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result


class ResumePolishExecutor(BaseExecutor):
    """简历润色执行器（支持流式）"""

    name = "resume_polish"
    description = "基于评分与建议对简历表述进行专业化润色改写"
    required_inputs = ["resume"]
    streamable = True

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from agents.skills.resume_agents.polish_agent import ResumePolishAgent

            agent = ResumePolishAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "score_result": None,
                "match_result": None,
                "suggestions": None,
                "overall_score": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            agent.run(state)

            polished = state.get("optimized_resume")
            if not polished:
                result.error = state.get("error") or "润色结果为空"
                return result

            result.success = True
            result.data = {"optimized_resume": polished}
            result.summary = f"润色完成，输出 {len(polished)} 字"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result

    def run_stream(self, task_input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """流式润色：逐块 yield {"type": "token"/"error", ...}"""
        error = self.validate(task_input)
        if error:
            yield {"type": "error", "content": error}
            return

        try:
            from agents.skills.resume_agents.polish_agent import ResumePolishAgent

            agent = ResumePolishAgent(llm=_resolve_llm(self._llm, task_input))
            state = {
                "resume": task_input["resume"],
                "jd": task_input.get("jd"),
                "score_result": None,
                "match_result": None,
                "suggestions": None,
                "overall_score": None,
                "optimized_resume": None,
                "error": None,
                "current_step": "started",
            }
            yield from agent.run_stream(state)
        except Exception as e:
            logger.error(f"[{self.name}] 流式执行失败: {e}")
            yield {"type": "error", "content": str(e)}

    def finalize(self, raw_text: str, original_resume: str = "", task_input: Optional[Dict[str, Any]] = None) -> str:
        """对累积的流式文本做最终清理（复用润色 agent 的清理逻辑）"""
        from agents.skills.resume_agents.polish_agent import ResumePolishAgent

        cleaned = ResumePolishAgent(
            llm=_resolve_llm(self._llm, task_input or {})
        )._clean_result(raw_text, original_resume)
        return cleaned or raw_text.strip()


class ResumeParseExecutor(BaseExecutor):
    """简历文件解析执行器"""

    name = "resume_parse"
    description = "从上传的简历文件（PDF/Word）中提取结构化信息"
    required_inputs = ["file_path"]

    def run(self, task_input: Dict[str, Any]) -> StepResult:
        result = self._new_result(task_input.get("step_id", ""))
        error = self.validate(task_input)
        if error:
            result.error = error
            return result

        try:
            from rag.document_processor import parse_resume

            data = parse_resume(task_input["file_path"])
            if not data:
                result.error = "简历解析结果为空"
                return result

            result.success = True
            result.data = dict(data)
            # 解析产物回填黑板，供下游执行器（评分/润色）作为 resume 输入
            skills = data.get("skills", "")
            projects = data.get("projects", "")
            text = "\n\n".join(part for part in [skills, projects] if part)
            if text:
                result.data["resume"] = text
            result.summary = f"解析完成，提取 {len(result.data)} 个字段"
            return result

        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            result.error = str(e)
            return result


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
