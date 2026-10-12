"""规则校验（确定性、零成本）

- 关键词覆盖率：ground_truth.must_contain 关键点在输出中的命中比例
- 结构完整性：ground_truth.structure 板块关键词在输出中的命中比例
- 禁忌校验：ground_truth.forbidden 出现即扣分（编造/空泛表达）
- 长度校验：是否落在 [min_length, max_length] 区间

权重按"实际配置了要求的维度"动态归一化：
- 语义部分（关键词 + 结构）合计权重 0.9，只在配置了要求的语义维度间均分；
  未配置任何语义要求时该部分权重为 0，不白送分（避免跨类别口径不一致）。
- 长度部分权重 0.1。
最终除以总权重归一化到 0-1，保证不同类别满分一致且不偏袒无要求维度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .dataset import EvalCase, GroundTruth

# 语义维度合计权重 / 长度权重
_SEMANTIC_WEIGHT = 0.9
_LENGTH_WEIGHT = 0.1


@dataclass
class RuleResult:
    """单条用例的规则评分结果"""
    keyword_coverage: float          # 0.0-1.0
    structure_score: float           # 0.0-1.0
    forbidden_hits: List[str] = field(default_factory=list)
    length_ok: bool = True
    # 综合规则通过率 0.0-1.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "keyword_coverage": round(self.keyword_coverage, 4),
            "structure_score": round(self.structure_score, 4),
            "forbidden_hits": self.forbidden_hits,
            "length_ok": self.length_ok,
            "score": round(self.score, 4),
        }


def _hit_ratio(keywords: List[str], text: str) -> float:
    """关键词命中率（关键词在文本中出现即算命中，大小写不敏感）。无要求时返回 0（由权重处理，不白送）"""
    if not keywords:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for kw in keywords if kw and kw.lower() in lowered)
    return hits / len(keywords)


def evaluate_rules(output: str, gt: GroundTruth) -> RuleResult:
    """对单条输出执行规则校验"""
    output = output or ""

    keyword_coverage = _hit_ratio(gt.must_contain, output)
    structure_score = _hit_ratio(gt.structure, output)

    forbidden_hits = [w for w in gt.forbidden if w and w in output]

    length_ok = gt.min_length <= len(output) <= gt.max_length

    # 只在配置了要求的语义维度间分配语义权重，避免空维度白送分
    semantic_dims: List[float] = []
    if gt.must_contain:
        semantic_dims.append(keyword_coverage)
    if gt.structure:
        semantic_dims.append(structure_score)
    semantic_weight = _SEMANTIC_WEIGHT if semantic_dims else 0.0
    semantic_avg = sum(semantic_dims) / len(semantic_dims) if semantic_dims else 0.0

    total_weight = semantic_weight + _LENGTH_WEIGHT
    weighted = semantic_avg * semantic_weight + (_LENGTH_WEIGHT if length_ok else 0.0)
    score = weighted / total_weight if total_weight > 0 else 0.0

    # 禁忌词每个 -0.1（最低 0）
    score -= 0.1 * len(forbidden_hits)
    score = max(0.0, min(1.0, score))

    return RuleResult(
        keyword_coverage=keyword_coverage,
        structure_score=structure_score,
        forbidden_hits=forbidden_hits,
        length_ok=length_ok,
        score=score,
    )


def evaluate_case(output: str, case: EvalCase) -> RuleResult:
    """便捷入口：对用例执行规则校验"""
    return evaluate_rules(output, case.ground_truth)
