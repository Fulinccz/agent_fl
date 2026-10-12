"""Agent 层 — 模型抽象、技能 Agent 与主从多智能体编排。

包含三个子系统：
- providers/   模型 Provider 抽象（local/ GGUF 与 cloud/ DeepSeek 物理隔离）
- skills/      技能 Agent（简历评分、JD 匹配、简历润色、对话图等）
- mas/         主从多智能体系统（意图识别 → Master 规划 → 执行器执行 → ReAct 反思）
"""

from .registry import get_agent

__all__ = ["get_agent"]
