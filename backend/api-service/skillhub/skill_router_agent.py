from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from logger import get_logger
from .keyword_matcher import KeywordMatcher
from .vector_router import VectorSkillRouter
from .llm_classifier import LLMIntentClassifier

logger = get_logger(__name__)


@dataclass
class RouteResult:
    skill: str
    confidence: float
    source: str
    reason: str
    params: Dict[str, Any]


class SkillRouterAgent:
    """Skill 路由 Agent

    三层路由架构：
    L1 KeywordMatcher: 关键词快速匹配，高置信直接返回
    L2 VectorSkillRouter: 向量粗筛，语义相似匹配
    L3 LLMIntentClassifier: 模型推理，复杂意图最终决策

    决策链：
    用户输入 → L1 → [置信度>=0.9? 返回] → L2 → [唯一高匹配? 返回] → L3 → 返回
    """

    def __init__(
        self,
        keyword_threshold: float = 0.9,
        vector_threshold: float = 0.3,
        llm_provider: str = "local",
        llm_model: Optional[str] = None,
        llm: Optional[Any] = None
    ):
        self.keyword_matcher = KeywordMatcher(high_confidence_threshold=keyword_threshold)
        self.vector_router = VectorSkillRouter(similarity_threshold=vector_threshold)
        self.llm_classifier = LLMIntentClassifier(provider=llm_provider, model=llm_model, llm=llm)

        self.keyword_threshold = keyword_threshold
        self.vector_threshold = vector_threshold

    def route(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> RouteResult:
        """
        路由用户输入到合适的 Skill

        Args:
            user_input: 用户原始输入
            context: 可选上下文（如是否有上传文件、历史对话等）

        Returns:
            RouteResult: 路由结果
        """
        logger.info(f"[SkillRouter] Routing: {user_input[:60]}")

        # ========== L1: 关键词匹配 ==========
        kw_result = self.keyword_matcher.match(user_input)

        if kw_result.confidence >= self.keyword_threshold:
            logger.info(f"[SkillRouter] L1 Keyword hit: {kw_result.skill} (confidence={kw_result.confidence:.2f})")
            return RouteResult(
                skill=kw_result.skill,
                confidence=kw_result.confidence,
                source="keyword",
                reason=f"关键词匹配: {', '.join(kw_result.matched_keywords[:3])}",
                params={}
            )

        # ========== L2: 向量粗筛 ==========
        vec_candidates = self.vector_router.search(user_input, top_k=5)

        if vec_candidates:
            confident_skill = self.vector_router.get_confident_skill(vec_candidates, threshold=self.vector_threshold)
            if confident_skill:
                best = [c for c in vec_candidates if c.skill == confident_skill][0]
                logger.info(f"[SkillRouter] L2 Vector hit: {confident_skill} (score={best.score:.3f})")
                return RouteResult(
                    skill=confident_skill,
                    confidence=best.score,
                    source="vector",
                    reason=f"向量匹配: {best.text[:30]}",
                    params={}
                )

        # ========== L3: LLM 推理 ==========
        logger.info(f"[SkillRouter] L3 LLM fallback, candidates={len(vec_candidates)}")
        llm_result = self.llm_classifier.classify(user_input, vec_candidates, context)

        intent = llm_result.get("intent", "chat-general")
        confidence = llm_result.get("confidence", 0.5)

        return RouteResult(
            skill=intent,
            confidence=confidence,
            source="llm",
            reason=llm_result.get("reason", "LLM推理"),
            params=llm_result.get("params", {})
        )

    def batch_route(self, inputs: List[str]) -> List[RouteResult]:
        """批量路由"""
        return [self.route(inp) for inp in inputs]
