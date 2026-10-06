from typing import Dict, Any, Optional, List, Tuple
from logger import get_logger

logger = get_logger(__name__)


class MatchResult:
    def __init__(self, skill: Optional[str], confidence: float, matched_keywords: List[str]):
        self.skill = skill
        self.confidence = confidence
        self.matched_keywords = matched_keywords


class KeywordMatcher:
    """L1: 关键词快速匹配层

    基于硬编码关键词规则，O(1) 复杂度快速命中高频明确意图。
    """

    SKILL_KEYWORDS: Dict[str, List[str]] = {
        "resume-polishing": [
            "润色", "优化", "改进", "修改", "调整", "美化",
            "大白话", "口语化", "专业", "话术", "重写", "改写",
            "polish", "improve", "optimize", "rewrite",
            "措辞", "表达", "描述", "写得不好", "写得更"
        ],
        "jd-keyword-match": [
            "匹配", "JD", "岗位", "招聘", "职位", "关键词",
            "ATS", "筛选", "匹配度", "适合", "符合", "差距",
            "match", "job description", "keyword", "position",
            "岗位要求", "职位要求", "招聘要求", "能投吗", "适合我吗"
        ],
        "resume-score": [
            "评分", "打分", "评估", "评价", "分数", "得分",
            "多少分", "怎么样", "如何", "水平", "质量",
            "score", "rate", "evaluate", "evaluation", "grade",
            "诊断", "分析", "检测", "竞争力", "合格吗"
        ],
        "resume-parse": [
            "解析", "提取", "抽取", "结构化", "识别", "要素",
            "parse", "extract", "analyze structure", "fields",
            "信息提取", "内容提取", "数据化", "PDF解析"
        ],
        "chat-general": [
            "你好", "在吗", "你能", "有什么", "介绍", "功能",
            "help", "怎么用", "说明", "指南", "建议", "咨询",
            "hello", "hi", "what can you", "how to"
        ]
    }

    EXACT_PATTERNS: Dict[str, List[str]] = {
        "resume-polishing": ["简历润色", "润色简历", "优化简历", "改简历"],
        "jd-keyword-match": ["JD匹配", "匹配JD", "岗位匹配", "简历匹配"],
        "resume-score": ["简历评分", "评分简历", "打分", "简历诊断"],
        "resume-parse": ["解析简历", "提取简历", "简历解析"],
    }

    def __init__(self, high_confidence_threshold: float = 0.9, min_confidence: float = 0.3):
        self.high_confidence_threshold = high_confidence_threshold
        self.min_confidence = min_confidence

    def match(self, user_input: str) -> MatchResult:
        user_input_lower = user_input.lower().strip()

        # 1. 精确模式匹配（最高优先级）
        for skill_name, patterns in self.EXACT_PATTERNS.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    return MatchResult(skill_name, 1.0, [pattern])

        # 2. 关键词频率统计
        skill_scores: Dict[str, Tuple[int, List[str]]] = {}

        for skill_name, keywords in self.SKILL_KEYWORDS.items():
            matched = []
            for kw in keywords:
                if kw.lower() in user_input_lower:
                    matched.append(kw)

            if matched:
                # 去重后的匹配关键词数
                unique_matches = len(set(m.lower() for m in matched))
                # 基础分 + 额外匹配加分（上限控制）
                score = 0.3 + min(unique_matches * 0.15, 0.6)
                skill_scores[skill_name] = (score, matched)

        if not skill_scores:
            return MatchResult(None, 0.0, [])

        # 3. 选择最高分
        best_skill = max(skill_scores.items(), key=lambda x: x[1][0])
        skill_name, (score, matched) = best_skill

        # 4. 如果有多个技能得分接近，降低置信度
        sorted_scores = sorted(skill_scores.values(), key=lambda x: x[0], reverse=True)
        if len(sorted_scores) >= 2:
            top1, top2 = sorted_scores[0][0], sorted_scores[1][0]
            if top1 - top2 < 0.15:
                score *= 0.7  # 竞争激烈的场景降低置信度

        return MatchResult(skill_name, min(score, 1.0), matched)

    def should_use_vector(self, result: MatchResult) -> bool:
        """判断是否需要进入向量粗筛层"""
        if result.confidence < self.min_confidence:
            return True
        if result.confidence < self.high_confidence_threshold:
            return True
        return False
