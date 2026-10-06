"""多智能体系统（Master-Slave）核心数据结构

定义意图识别结果、任务计划、步骤结果、ReAct 反思结果等状态对象。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============ 意图名称常量 ============
INTENT_RESUME_OPTIMIZE = "resume-optimize"     # 简历全流程优化（复杂，需规划）
INTENT_RESUME_SCORE = "resume-score"           # 简历评分
INTENT_RESUME_POLISH = "resume-polishing"      # 简历润色
INTENT_JD_MATCH = "jd-keyword-match"           # JD 匹配分析
INTENT_RESUME_PARSE = "resume-parse"           # 简历解析
INTENT_CHAT_GENERAL = "chat-general"           # 通用对话
INTENT_COMPLEX_TASK = "complex-task"           # 复合任务（LLM 分解规划）


@dataclass
class IntentResult:
    """意图识别结果"""
    name: str                          # 意图名称（见上方常量）
    confidence: float = 0.0            # 置信度 0-1
    source: str = "rule"               # 识别来源: preset/rule/keyword/vector/llm
    reason: str = ""                   # 识别理由
    params: Dict[str, Any] = field(default_factory=dict)  # 附加参数
    is_complex: bool = False           # 是否为复杂任务（需要 master 规划）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "reason": self.reason,
            "is_complex": self.is_complex,
        }


# ============ 步骤状态常量 ============
STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_SUCCESS = "success"
STEP_FAILED = "failed"
STEP_SKIPPED = "skipped"


@dataclass
class TaskStep:
    """任务计划中的单个步骤（slave 执行单元）"""
    step_id: str                                       # 如 "step_1"
    agent: str                                         # 执行器名称（见 executors.py）
    task: str                                          # 自然语言任务描述
    depends_on: List[str] = field(default_factory=list)  # 依赖的前置步骤 id
    params: Dict[str, Any] = field(default_factory=dict)  # 步骤附加参数
    status: str = STEP_PENDING
    retries: int = 0                                   # 已重试次数
    result: Optional["StepResult"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "task": self.task,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "retries": self.retries,
        }


@dataclass
class TaskPlan:
    """Master 产出的任务计划"""
    goal: str                                          # 总目标
    intent: str                                        # 关联意图
    steps: List[TaskStep] = field(default_factory=list)
    source: str = "template"                           # template / llm / replan

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "intent": self.intent,
            "source": self.source,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class StepResult:
    """执行器执行结果"""
    step_id: str = ""
    agent: str = ""
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)   # 结构化输出
    summary: str = ""                                    # 供 master/反思阅读的简短摘要
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "success": self.success,
            "summary": self.summary,
            "error": self.error,
            "data": self.data,
        }


# ============ ReAct 反思裁决常量 ============
VERDICT_CONTINUE = "continue"   # 结果合格，继续下一步
VERDICT_RETRY = "retry"         # 结果不合格，重试当前步骤
VERDICT_REPLAN = "replan"       # 多次失败，重新规划剩余步骤
VERDICT_FINISH = "finish"       # 目标已达成，可提前结束
VERDICT_ABORT = "abort"         # 无法继续，终止流程


@dataclass
class Reflection:
    """ReAct 反思结果"""
    step_id: str = ""
    verdict: str = VERDICT_CONTINUE
    thought: str = ""            # ReAct Thought
    feedback: str = ""           # 重试时给执行器的改进反馈
    source: str = "rule"         # rule / llm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "verdict": self.verdict,
            "thought": self.thought,
            "feedback": self.feedback,
            "source": self.source,
        }


class Blackboard:
    """共享黑板：master 与各执行器之间的数据总线

    - 输入区：用户消息、简历文本、JD、RAG 上下文等
    - 结果区：各步骤执行产出的结构化数据（按 step_id / agent 双索引）
    """

    def __init__(self, **inputs: Any):
        self.inputs: Dict[str, Any] = dict(inputs)
        self.results: Dict[str, StepResult] = {}          # step_id -> result
        self.agent_results: Dict[str, StepResult] = {}    # agent -> 最近一次 result

    def set_input(self, key: str, value: Any):
        self.inputs[key] = value

    def record(self, result: StepResult):
        self.results[result.step_id] = result
        self.agent_results[result.agent] = result

    def snapshot(self) -> Dict[str, Any]:
        """合并输入与所有成功结果，供执行器取用"""
        merged = dict(self.inputs)
        for result in self.results.values():
            if result.success:
                merged.update(result.data)
        return merged

    def trace_lines(self) -> List[str]:
        """生成 ReAct Observation 轨迹文本（供反思/重规划提示词使用）"""
        lines = []
        for step_id, result in self.results.items():
            status = "成功" if result.success else f"失败({result.error[:80]})"
            lines.append(f"[{step_id}|{result.agent}] {status} {result.summary[:200]}")
        return lines
