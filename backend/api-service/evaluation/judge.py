"""LLM-as-Judge 语义评分

用更强的模型作为"裁判"，对输出按三个维度评分：
- correctness  正确性（1-5）
- relevance    相关性（1-5）
- has_hallucination 是否幻觉（编造简历中不存在的经历/技能/事实）

输出 JSON；解析失败自动重试 1 次，仍失败则 available=False，由 runner 回退到纯规则评分。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict

from logger import get_logger
from .dataset import EvalCase

logger = get_logger(__name__)


JUDGE_PROMPT_TEMPLATE = """你是一个严格的简历优化 Agent 输出评审专家。请根据【任务要求】、【输入简历】和【模型输出】评分。

【任务类型】{category}
{jd_block}
【输入简历】
{resume}

【模型输出】
{output}

请只输出一个 JSON 对象，不要输出任何其他内容，格式如下：
{{
  "correctness": <1到5的整数，输出是否正确完成任务、有无事实错误>,
  "relevance": <1到5的整数，输出与简历和任务的相关程度>,
  "has_hallucination": <true或false，是否编造了输入简历中不存在的经历、技能、公司、数据等>,
  "reason": "<不超过50字的中文评语>"
}}

评分标准：5=优秀 4=良好 3=合格 2=较差 1=很差。"""


@dataclass
class JudgeScore:
    correctness: int = 0
    relevance: int = 0
    has_hallucination: bool = False
    reason: str = ""
    available: bool = False

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "has_hallucination": self.has_hallucination,
            "reason": self.reason,
            "available": self.available,
        }


# Judge 调用函数签名：(prompt: str) -> str
JudgeFn = Callable[[str], str]

# 裁判 Agent 缓存（按 provider+model 复用，避免每条用例重建 HTTP 客户端）
_judge_agent_cache: Dict[tuple, Any] = {}


def _get_cached_judge_agent(judge_provider: str, model: Optional[str]):
    key = (judge_provider, model)
    if key not in _judge_agent_cache:
        # 评测为用户显式发起的离线跑批，直连 registry，绕过线上 Provider 的成本开关
        # （get_shared_llm 在 deepseek 未启用时会静默回落 local，导致对比结果失真）
        from agents.registry import get_agent
        _judge_agent_cache[key] = get_agent(provider=judge_provider, model=model)
    return _judge_agent_cache[key]


def _default_judge_fn(judge_provider: str = "deepseek", model: Optional[str] = None) -> JudgeFn:
    """默认裁判：通过 registry 获取并缓存 Provider（测试时可注入 judge_fn 替换）"""
    def _call(prompt: str) -> str:
        return _get_cached_judge_agent(judge_provider, model).generate(prompt)
    return _call


def _parse_json(raw: str) -> Optional[dict]:
    """从容错文本中提取 JSON 对象"""
    if not raw:
        return None
    # 优先提取 ```json fenced block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # 退而求其次：第一个 { 到最后一个 }
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def judge_output(
    output: str,
    case: EvalCase,
    judge_fn: Optional[JudgeFn] = None,
    judge_provider: str = "deepseek",
    judge_model: Optional[str] = None,
) -> JudgeScore:
    """对单条输出执行 LLM 评审"""
    if judge_fn is None:
        judge_fn = _default_judge_fn(judge_provider, judge_model)

    jd_block = f"【岗位 JD】\n{case.jd}\n\n" if case.jd else ""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        category=case.category,
        jd_block=jd_block,
        resume=case.input[:4000],
        output=(output or "")[:4000],
    )

    raw: Optional[str] = None
    parsed: Optional[dict] = None
    # 最多尝试 2 次
    for attempt in range(2):
        try:
            raw = judge_fn(prompt)
            parsed = _parse_json(raw)
            if parsed is not None:
                break
        except Exception as e:
            logger.warning("LLM Judge 调用失败（第%d次）：%s", attempt + 1, e)

    if not parsed:
        logger.warning("用例 %s 的 Judge 输出解析失败，回退纯规则评分", case.id)
        return JudgeScore(available=False)

    def _clamp_int(v: Any, default: int = 3) -> int:
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return default

    return JudgeScore(
        correctness=_clamp_int(parsed.get("correctness")),
        relevance=_clamp_int(parsed.get("relevance")),
        has_hallucination=bool(parsed.get("has_hallucination", False)),
        reason=str(parsed.get("reason", ""))[:200],
        available=True,
    )
