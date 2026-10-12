"""Agent 评测框架

双轨评分：
- 规则校验（rules.py）：关键词覆盖率 / 结构完整性 / 禁忌词，确定性零成本
- LLM-as-Judge（judge.py）：正确性 / 相关性 / 幻觉，语义评分

能力：
- 同一评测集对多个 Provider（local / deepseek）跑批对比
- 指标：准确率、幻觉率、相关性、平均延迟、估算 token
- 基线回归：保存基线报告，后续跑分对比，指标退化超阈值即失败（可接 CI）
"""
from .dataset import EvalCase, EvalDataset, build_prompt
from .rules import RuleResult, evaluate_rules
from .judge import JudgeScore, judge_output
from .runner import ProviderReport, run_evaluation, compare_to_baseline

__all__ = [
    "EvalCase",
    "EvalDataset",
    "build_prompt",
    "RuleResult",
    "evaluate_rules",
    "JudgeScore",
    "judge_output",
    "ProviderReport",
    "run_evaluation",
    "compare_to_baseline",
]
