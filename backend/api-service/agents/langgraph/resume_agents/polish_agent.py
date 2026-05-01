"""简历润色 Agent"""
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Generator

from agents.registry import get_agent
from logger import get_logger
from .state import ResumeState

logger = get_logger(__name__)


@dataclass
class ResumePolishAgent:
    """简历润色 Agent - 优化简历表述"""

    name: str = "resume_polish_agent"
    llm: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        if self.llm is None:
            self.llm = get_agent(provider="local")

    def run(self, state: ResumeState) -> ResumeState:
        """执行简历润色（非流式）"""
        logger.info(f"[{self.name}] 开始简历润色")

        try:
            resume = state["resume"]
            jd = state.get("jd")

            prompt = self._build_prompt(resume, jd)
            result = self.llm.generate(prompt, deepThinking=False, max_new_tokens=1024)

            logger.debug(f"[{self.name}] 原始结果: {result[:500]}...")

            polished = self._clean_result(result, resume)

            if not polished or len(polished) < 50:
                logger.warning(f"[{self.name}] 清理后内容为空或太短，使用原始简历")
                polished = resume[:2000]

            state["optimized_resume"] = polished
            state["current_step"] = "polish_completed"

            logger.info(f"[{self.name}] 润色完成，原始长度: {len(result)}, 清理后长度: {len(polished)}")

        except Exception as e:
            logger.error(f"[{self.name}] 润色失败: {e}")
            state["error"] = f"润色失败: {str(e)}"
            state["optimized_resume"] = state["resume"]

        return state

    def run_stream(self, state: ResumeState) -> Generator[Dict[str, Any], None, None]:
        """流式执行简历润色，逐token生成"""
        logger.info(f"[{self.name}] 开始流式简历润色")

        try:
            resume = state["resume"]
            jd = state.get("jd")

            prompt = self._build_prompt(resume, jd)

            for chunk in self.llm.generate_stream(prompt, deepThinking=False, max_new_tokens=1024):
                yield chunk

        except Exception as e:
            logger.error(f"[{self.name}] 流式润色失败: {e}")
            yield {"type": "error", "message": str(e)}

    def _build_prompt(self, resume: str, jd: Optional[str] = None) -> str:
        """构建润色prompt - 简化版，适合小模型"""
        jd_hint = f"目标岗位：{jd[:500]}\n\n" if jd else ""

        return f"""{jd_hint}请优化以下简历描述，使其更专业：

{resume[:3000]}

优化要求：
- 用"主导"、"设计"、"优化"替换"做"、"负责"
- 添加数字，如"提升30%"、"完成5个"
- 突出技术难点

重要：只输出优化后的简历内容，不要添加任何标题、说明、角色标记（如user/assistant）或额外内容。
直接输出优化后的描述："""

    def _clean_result(self, result: str, original_resume: str = "") -> str:
        """清理润色结果，移除思考标签和常见前缀"""
        import re
        import traceback

        # 记录调用栈
        stack = traceback.extract_stack()
        caller = stack[-2] if len(stack) >= 2 else None
        caller_info = f"{caller.filename.split('/')[-1]}:{caller.lineno}" if caller else "unknown"
        
        # 记录原始输出用于调试
        logger.info(f"[_clean_result] 被 {caller_info} 调用")
        logger.info(f"[_clean_result] 原始输出前200字符: {result[:200]!r}")
        logger.info(f"[_clean_result] 原始输出长度: {len(result)}")

        # 移除思考标签及其内容
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)

        # 移除代码块标记
        result = re.sub(r'```\w*\n?', '', result)
        result = re.sub(r'```', '', result)

        # 移除Markdown标记
        result = re.sub(r'\*\*', '', result)
        result = re.sub(r'###\s*', '', result)
        result = re.sub(r'##\s*', '', result)
        result = re.sub(r'#\s*', '', result)

        # 移除 assistant 标记及其后面的内容（但保留 assistant 标记后的内容）
        # 只移除 "assistant:" 这个标记本身，不移除后面的内容
        result = re.sub(r'^assistant[:：]\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\nassistant[:：]\s*', '\n', result, flags=re.IGNORECASE)

        # 记录中间结果
        logger.info(f"[_clean_result] 基础清理后长度: {len(result)}")

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

        # 移除角色标记及其后面的所有内容（仅当角色标记是独立段落时）
        result = re.sub(r'\n\s*(user|assistant|system|human|ai)[:：].*', '', result, flags=re.DOTALL | re.IGNORECASE)

        # 移除结尾的 user/assistant 标记（不带冒号的情况）
        result = re.sub(r'\s*\b(user|assistant|system|human|ai)\b\s*$', '', result, flags=re.IGNORECASE)

        # 移除多余的空行
        result = re.sub(r'\n{3,}', '\n\n', result)

        # 记录最终结果
        logger.info(f"[_clean_result] 最终清理后长度: {len(result)}")
        logger.info(f"[_clean_result] 最终结果前200字符: {result[:200]!r}")

        # 如果结果太短，可能是被过度过滤了，返回原始简历
        if len(result.strip()) < 50 and len(original_resume) > 100:
            logger.warning(f"清理后内容太短({len(result)}字符)，返回原始简历前1000字符")
            logger.warning(f"原始输出内容: {result[:500]!r}")
            return original_resume[:1000]

        return result.strip()
