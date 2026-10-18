"""Master-Slave 多智能体系统

架构（ReAct 与 Reflection 职责分离）：
    用户输入 → IntentRecognizer（意图识别）
             → MasterAgent（任务规划）
             → Executor（执行器，Slave 层；可选 ReAct 步内回路 Thought→Action→Observation）
             → Reflector（步后评审：RuleCritic 规则快评 + LLMCritic 深审，只评估不控制）
             → MasterOrchestrator（控制策略唯一入口：重试预算/重规划/跳过 + 事件流输出）
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
from .reflector import Reflector, RuleCritic, LLMCritic
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
    'get_shared_llm', 'IntentRecognizer', 'MasterAgent',
    'Reflector', 'RuleCritic', 'LLMCritic',
    # 执行器
    'BaseExecutor', 'ExecutorRegistry', 'ResumeScoreExecutor', 'JDMatchExecutor',
    'ResumePolishExecutor', 'ResumeParseExecutor', 'ChatExecutor',
    # 编排器
    'MasterOrchestrator',
]
