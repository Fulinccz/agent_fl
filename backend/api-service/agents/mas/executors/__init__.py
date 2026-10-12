"""执行器子包 — re-export 全部符号

每个执行器封装一项原子能力，接受黑板快照 + 步骤参数作为输入，
返回统一的 StepResult。Master 只依赖执行器名称与输入输出契约。

内置执行器：
- resume_score   简历评分        （复用 ResumeScoreAgent）
- jd_match       JD 匹配分析     （复用 JDMatchAgent）
- resume_polish  简历润色        （复用 ResumePolishAgent，支持流式）
- resume_parse   简历文件解析    （复用 rag.document_processor）
- chat           通用对话        （直接调用共享 LLM，支持流式）
"""
from .base import BaseExecutor, _resolve_llm
from .resume_score import ResumeScoreExecutor
from .jd_match import JDMatchExecutor
from .resume_polish import ResumePolishExecutor
from .resume_parse import ResumeParseExecutor
from .resume_condense import ResumeCondenseExecutor
from .chat import ChatExecutor
from .registry import ExecutorRegistry

__all__ = [
    'BaseExecutor',
    '_resolve_llm',
    'ResumeScoreExecutor',
    'JDMatchExecutor',
    'ResumePolishExecutor',
    'ResumeParseExecutor',
    'ResumeCondenseExecutor',
    'ChatExecutor',
    'ExecutorRegistry',
]
