"""评测集定义与加载"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

# 评测任务类别
CATEGORY_SCORE = "resume_score"
CATEGORY_MATCH = "jd_match"
CATEGORY_POLISH = "resume_polish"
VALID_CATEGORIES = {CATEGORY_SCORE, CATEGORY_MATCH, CATEGORY_POLISH}

# 默认评测集
DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "eval_samples.json"


@dataclass
class GroundTruth:
    """标准答案 / 校验规则"""
    must_contain: List[str] = field(default_factory=list)   # 输出应包含的关键点
    structure: List[str] = field(default_factory=list)      # 输出应保留的结构板块
    forbidden: List[str] = field(default_factory=list)      # 输出不应出现的词（编造/空泛）
    min_length: int = 0
    max_length: int = 100000

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroundTruth":
        return cls(
            must_contain=d.get("must_contain", []),
            structure=d.get("structure", []),
            forbidden=d.get("forbidden", []),
            min_length=int(d.get("min_length", 0)),
            max_length=int(d.get("max_length", 100000)),
        )


@dataclass
class EvalCase:
    """单条评测用例"""
    id: str
    category: str
    input: str
    jd: Optional[str] = None
    instruction: Optional[str] = None   # 可覆盖按类别生成的默认指令
    ground_truth: GroundTruth = field(default_factory=GroundTruth)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalCase":
        category = d.get("category", "")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"评测用例 {d.get('id')} 的 category 非法：{category}")
        return cls(
            id=d["id"],
            category=category,
            input=d["input"],
            jd=d.get("jd"),
            instruction=d.get("instruction"),
            ground_truth=GroundTruth.from_dict(d.get("ground_truth", {})),
        )


class EvalDataset:
    """评测集"""

    def __init__(self, cases: List[EvalCase]):
        if not cases:
            raise ValueError("评测集为空")
        self.cases = cases

    @classmethod
    def from_json(cls, path: Optional[str | Path] = None) -> "EvalDataset":
        path = Path(path) if path else DEFAULT_DATASET_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 兼容两种格式：顶层数组 或 {"cases": [...]}
        items = raw["cases"] if isinstance(raw, dict) else raw
        return cls([EvalCase.from_dict(item) for item in items])

    def filter(self, category: Optional[str] = None) -> "EvalDataset":
        if not category:
            return self
        return EvalDataset([c for c in self.cases if c.category == category])

    def __len__(self) -> int:
        return len(self.cases)


def build_prompt(case: EvalCase) -> str:
    """根据任务类别构造统一评测指令（所有 Provider 使用同一 prompt，保证对比公平）"""
    if case.instruction:
        base = case.instruction
    elif case.category == CATEGORY_POLISH:
        base = "请优化以下简历内容，使其更专业、更具竞争力，保留真实经历，不要编造："
    elif case.category == CATEGORY_SCORE:
        base = "请对以下简历进行多维度评分并给出改进建议："
    elif case.category == CATEGORY_MATCH:
        base = "请分析以下简历与目标岗位 JD 的匹配度，指出匹配点与差距："
    else:
        base = "请处理以下简历内容："

    prompt = f"{base}\n\n【简历】\n{case.input}"
    if case.category == CATEGORY_MATCH:
        prompt += f"\n\n【岗位 JD】\n{case.jd or '（未提供）'}"
    elif case.jd:
        prompt += f"\n\n【目标岗位 JD】\n{case.jd}"
    return prompt
