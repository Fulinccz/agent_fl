"""简历润色 Agent"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Generator, List

from agents.registry import get_agent
from logger import get_logger
from services.tracing import start_span, update_observation
from .state import ResumeState
from .textutil import split_text, strip_llm_artifacts

logger = get_logger(__name__)

# 单次润色输入上限：超过则分段逐块润色（避免 prompt 截断丢内容 / 输出超 max_tokens 被截断）
POLISH_SEGMENT_CHARS = 2400


@dataclass
class ResumePolishAgent:
    """简历润色 Agent - 优化简历表述"""

    name: str = "resume_polish_agent"
    llm: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        if self.llm is None:
            self.llm = get_agent(provider="local")

    def run(self, state: ResumeState) -> ResumeState:
        """执行简历润色（非流式，长文分段后拼接）"""
        logger.info(f"[{self.name}] 开始简历润色")

        try:
            resume = state["resume"]
            jd = state.get("jd")
            extra = state.get("extra_instruction")

            segments = split_text(resume, POLISH_SEGMENT_CHARS) if len(resume) > POLISH_SEGMENT_CHARS else [resume]
            parts: List[str] = []
            for idx, segment in enumerate(segments):
                prompt = self._build_prompt(
                    segment, jd, extra,
                    part_hint=self._part_hint(idx, len(segments)),
                )
                with start_span("llm-polish", as_type="generation", input=prompt[:500]) as obs:
                    result = self.llm.generate(prompt, deepThinking=False, max_new_tokens=1024)
                    update_observation(obs, output=result[:500])
                parts.append(result)

            polished_raw = "\n\n".join(parts)
            polished = self._clean_result(polished_raw, resume)
            polished = self._format_output(polished)

            if not polished or len(polished) < 50:
                logger.warning(f"[{self.name}] 清理后内容为空或太短，使用原始简历")
                polished = resume[:2000]

            state["optimized_resume"] = polished
            state["current_step"] = "polish_completed"

            logger.info(
                f"[{self.name}] 润色完成（{len(segments)} 段），"
                f"原始长度: {len(resume)}, 清理后长度: {len(polished)}"
            )

        except Exception as e:
            logger.error(f"[{self.name}] 润色失败: {e}")
            state["error"] = f"润色失败: {str(e)}"
            state["optimized_resume"] = state["resume"]

        return state

    def run_stream(self, state: ResumeState) -> Generator[Dict[str, Any], None, None]:
        """流式执行简历润色：长文分段逐块流式输出，段间补换行"""
        logger.info(f"[{self.name}] 开始流式简历润色")

        try:
            resume = state["resume"]
            jd = state.get("jd")
            extra = state.get("extra_instruction")

            segments = split_text(resume, POLISH_SEGMENT_CHARS) if len(resume) > POLISH_SEGMENT_CHARS else [resume]
            for idx, segment in enumerate(segments):
                if idx > 0:
                    # 分段之间保留段落间隔
                    yield {"type": "token", "content": "\n\n"}

                prompt = self._build_prompt(
                    segment, jd, extra,
                    part_hint=self._part_hint(idx, len(segments)),
                )

                with start_span("llm-polish-stream", as_type="generation", input=prompt[:500]) as obs:
                    accumulated = ""
                    for chunk in self.llm.generate_stream(prompt, deepThinking=False, max_new_tokens=1024):
                        if chunk.get("type") == "token":
                            accumulated += chunk.get("content", "")
                        yield chunk
                    update_observation(obs, output=accumulated[:500])

            logger.info(f"[{self.name}] 流式润色完成（{len(segments)} 段）")

        except Exception as e:
            logger.error(f"[{self.name}] 流式润色失败: {e}")
            # 错误事件键与 provider/master 契约统一为 content
            yield {"type": "error", "content": str(e)}

    @staticmethod
    def _part_hint(idx: int, total: int) -> str:
        if total <= 1:
            return ""
        return (
            f"\n注意：这是简历的第 {idx + 1}/{total} 部分，"
            "只优化本部分内容，保留分节标题，不要添加总起、总结或重复个人信息。"
        )

    def _build_prompt(
        self,
        resume: str,
        jd: Optional[str] = None,
        extra_instruction: Optional[str] = None,
        part_hint: str = "",
    ) -> str:
        """构建润色 prompt - 简化版，适合小模型（输入应为已切分的单段）"""
        jd_hint = f"目标岗位：{jd[:500]}\n\n" if jd else ""
        extra_hint = f"\n用户特别要求：{extra_instruction[:300]}\n" if extra_instruction else ""

        return f"""{jd_hint}请优化以下简历描述，使其更专业：

{resume}{extra_hint}{part_hint}

优化要求：
- 用"主导"、"设计"、"优化"替换"做"、"负责"
- 添加数字，如"提升30%"、"完成5个"
- 突出技术难点

重要：只输出优化后的简历内容，不要添加任何标题、说明、角色标记（如user/assistant）或额外内容。
直接输出优化后的描述："""

    def _clean_result(self, result: str, original_resume: str = "") -> str:
        """清理润色结果：think/围栏（公共内核）+ markdown 标记 + 角色标记"""
        import re

        # 公共内核：移除思考标签及其内容、代码块标记
        result = strip_llm_artifacts(result)

        # 移除Markdown标记
        result = re.sub(r'\*\*', '', result)
        result = re.sub(r'###\s*', '', result)
        result = re.sub(r'##\s*', '', result)
        result = re.sub(r'#\s*', '', result)

        # 移除 assistant 标记及其后面的内容（但保留 assistant 标记后的内容）
        # 只移除 "assistant:" 这个标记本身，不移除后面的内容
        result = re.sub(r'^assistant[:：]\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\nassistant[:：]\s*', '\n', result, flags=re.IGNORECASE)

        # 移除明显的角色标记行（如 "user:" 或 "assistant:" 开头的新段落）
        # 只移除这些标记行本身，不截断后续内容
        lines = result.split('\n')
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            # 跳过纯角色标记行（如单独的 "user:" 或 "assistant:"）
            if re.match(r'^(user|assistant|system|human|ai)[:：]?\s*$', line_stripped, re.IGNORECASE):
                logger.debug(f"检测到角色标记，移除该行: {line_stripped[:50]}")
                continue
            filtered_lines.append(line)
        result = '\n'.join(filtered_lines)

        # 移除角色标记独立段落（该行起整段删除，仅限标记行本身独立成段）
        result = re.sub(r'\n\s*(user|assistant|system|human|ai)[:：][^\n]*', '', result, flags=re.IGNORECASE)

        # 移除结尾残留的 user/assistant 标记词（仅当是最后一个词且为标记语义）
        result = re.sub(r'\s+(user|assistant|system)\s*$', '', result, flags=re.IGNORECASE)

        # 移除多余的空行
        result = re.sub(r'\n{3,}', '\n\n', result)

        # 如果结果太短，可能是被过度过滤了，返回原始简历
        if len(result.strip()) < 50 and len(original_resume) > 100:
            logger.warning(
                f"清理后内容过短({len(result.strip())}字符)，回退原始简历前1000字符"
            )
            return original_resume[:1000]

        return result.strip()

    @staticmethod
    def _format_output(text: str) -> str:
        """通用输出排版归一化（复用 OutputFormatter skill，失败时静默降级）"""
        try:
            from agents.skills.output_formatter import OutputFormatter
            return OutputFormatter().format(text) or text
        except Exception as e:
            logger.warning(f"输出格式化失败，跳过: {e}")
            return text
