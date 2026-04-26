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

直接输出优化后的描述："""

    def _clean_result(self, result: str, original_resume: str = "") -> str:
        """清理润色结果，移除思考标签和常见前缀"""
        import re

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

        # 移除 assistant 标记及其后面的内容
        result = re.sub(r'assistant[:：].*', '', result, flags=re.DOTALL | re.IGNORECASE)

        # 检测并移除重复内容（如"保持原意不变"的循环）
        # 如果同一句话重复出现3次以上，截断到第一次出现的位置
        lines = result.split('\n')
        seen_lines = set()
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # 检测重复行（忽略数字差异）
            line_normalized = re.sub(r'\d+', 'N', line_stripped)
            if line_normalized in seen_lines:
                # 发现重复，停止处理
                break
            seen_lines.add(line_normalized)
            filtered_lines.append(line)
        result = '\n'.join(filtered_lines)

        # 移除模板内容行（以特定关键词开头的行）
        skip_keywords = [
            '岗位：', '岗位:', '项目经历：', '项目经历:',
            '个人简历', '姓名：', '姓名:', '联系电话：', '联系电话:',
            '电子邮箱：', '电子邮箱:', '求职意向：', '求职意向:',
            '要求：', '要求:', '改写示例', '原始：', '改写：'
        ]
        lines = result.split('\n')
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            # 如果行以跳过关键词开头，则跳过该行及之后所有内容
            if any(line_stripped.startswith(keyword) for keyword in skip_keywords):
                break
            filtered_lines.append(line)
        result = '\n'.join(filtered_lines)

        # 移除多余的空行
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result.strip()
