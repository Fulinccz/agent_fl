"""简历文件解析执行器"""
from __future__ import annotations

from typing import Any, Dict

from logger import get_logger
from .base import BaseExecutor
from ..state import StepResult

logger = get_logger(__name__)


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
