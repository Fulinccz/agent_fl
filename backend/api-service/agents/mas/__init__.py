"""Master-Slave 多智能体系统

架构：
    用户输入 → IntentRecognizer（意图识别）
             → MasterAgent（任务规划）
             → ExecutorRegistry（执行器派发，Slave 层）
             → Reflector（ReAct 反思：continue/retry/replan/finish/abort）
             → MasterOrchestrator（编排与事件流输出）
"""
from .state import (
    IntentResult,
    TaskPlan,
    TaskStep,
    StepResult,
    Reflection,
    Blackboard,
    INTENT_RESUME_OPTIMIZE,
    INTENT_RESUME_SCORE,
    INTENT_RESUME_POLISH,
    INTENT_JD_MATCH,
    INTENT_RESUME_PARSE,
    INTENT_CHAT_GENERAL,
    INTENT_COMPLEX_TASK,
)
from agents.llm import get_shared_llm
from .intent import IntentRecognizer
from .planner import MasterAgent
from .reflector import Reflector
from .executors import (
    BaseExecutor,
    ExecutorRegistry,
    ResumeScoreExecutor,
    JDMatchExecutor,
    ResumePolishExecutor,
    ResumeParseExecutor,
    ChatExecutor,
)
from .master import MasterOrchestrator

__all__ = [
    # 状态
    'IntentResult', 'TaskPlan', 'TaskStep', 'StepResult', 'Reflection', 'Blackboard',
    'INTENT_RESUME_OPTIMIZE', 'INTENT_RESUME_SCORE', 'INTENT_RESUME_POLISH',
    'INTENT_JD_MATCH', 'INTENT_RESUME_PARSE', 'INTENT_CHAT_GENERAL', 'INTENT_COMPLEX_TASK',
    # 组件
    'get_shared_llm', 'IntentRecognizer', 'MasterAgent', 'Reflector',
    # 执行器
    'BaseExecutor', 'ExecutorRegistry', 'ResumeScoreExecutor', 'JDMatchExecutor',
    'ResumePolishExecutor', 'ResumeParseExecutor', 'ChatExecutor',
    # 编排器
    'MasterOrchestrator',
]
