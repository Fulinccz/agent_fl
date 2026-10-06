"""意图识别模块

复用 skill_router 的三层路由引擎（L1 关键词 → L2 向量 → L3 LLM 推理），
在其上增加：
1. 预置意图直通（resume 路由等已知场景跳过识别）
2. 复杂度判定（是否需要 Master 规划多步骤）
3. resume-optimize 复合意图识别（评分+匹配+润色多步任务）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from logger import get_logger
from .state import (
    IntentResult,
    INTENT_RESUME_OPTIMIZE,
    INTENT_CHAT_GENERAL,
)

logger = get_logger(__name__)

# 显式多步优化信号词（命中即视为复杂任务）
_COMPLEX_MARKERS = [
    "一键优化", "全程优化", "全面优化", "彻底优化", "完整优化",
    "优化简历", "简历优化", "帮我优化", "优化这份简历", "优化一下简历",
    "评分并润色", "评分和润色", "润色并评分",
    "optimize resume", "full optimize",
]

# 单技能意图 → 执行器映射（简单任务直接派发，无需规划）
_SIMPLE_SKILL_AGENTS: Dict[str, str] = {
    "resume-score": "resume_score",
    "resume-polishing": "resume_polish",
    "jd-keyword-match": "jd_match",
    "resume-parse": "resume_parse",
}


class IntentRecognizer:
    """意图识别器（Master 的入口哨兵）

    决策流：
    1. preset 预置意图直接返回
    2. 规则预判：命中复杂任务信号词 → resume-optimize
    3. 三层路由（关键词/向量/LLM）得到单技能意图
    4. 后处理：结合上下文（是否有简历/JD）修正复杂度
    """

    def __init__(self, llm: Optional[Any] = None, enable_vector: bool = True):
        self._llm = llm
        self._enable_vector = enable_vector
        self._router = None  # 延迟初始化（向量层依赖 faiss 索引，可能缺失）

    @property
    def router(self):
        """延迟加载路由器，加载失败时降级为 None（纯规则识别）

        enable_vector=True  → 完整三层路由（关键词/向量/LLM）
        enable_vector=False → 轻量路由（关键词/LLM，不加载向量检索模型）
        """
        if self._router is None:
            try:
                if self._enable_vector:
                    from skillhub.skill_router_agent import SkillRouterAgent
                    self._router = SkillRouterAgent(llm=self._llm)
                else:
                    self._router = self._build_lite_router()
                logger.info("[IntentRecognizer] 意图路由器初始化完成 (vector=%s)", self._enable_vector)
            except Exception as e:
                logger.warning(f"[IntentRecognizer] 路由器初始化失败，降级为规则识别: {e}")
        return self._router

    def _build_lite_router(self):
        """构建轻量路由器：关键词层 + LLM 层（跳过向量粗筛）"""
        from skillhub.keyword_matcher import KeywordMatcher
        from skillhub.llm_classifier import LLMIntentClassifier
        from skillhub.skill_router_agent import RouteResult

        keyword_matcher = KeywordMatcher()
        llm_classifier = LLMIntentClassifier(llm=self._llm)

        class _LiteRouter:
            def route(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> RouteResult:
                kw = keyword_matcher.match(user_input)
                if kw.confidence >= keyword_matcher.high_confidence_threshold:
                    return RouteResult(
                        skill=kw.skill, confidence=kw.confidence, source="keyword",
                        reason=f"关键词匹配: {', '.join(kw.matched_keywords[:3])}", params={},
                    )
                result = llm_classifier.classify(user_input, [], context)
                return RouteResult(
                    skill=result.get("intent", INTENT_CHAT_GENERAL),
                    confidence=result.get("confidence", 0.5),
                    source="llm",
                    reason=result.get("reason", "LLM推理"),
                    params=result.get("params", {}),
                )

        return _LiteRouter()

    def recognize(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        preset: Optional[IntentResult] = None,
    ) -> IntentResult:
        """识别用户意图

        Args:
            user_input: 用户原始输入
            context: 上下文（resume/jd/file_path/rag_context 等）
            preset: 预置意图（非空则直接返回）
        """
        context = context or {}

        # 1. 预置意图直通
        if preset is not None:
            logger.info(f"[IntentRecognizer] 使用预置意图: {preset.name}")
            return preset

        has_resume = bool(context.get("resume"))
        has_jd = bool(context.get("jd"))
        has_file = bool(context.get("file_path"))

        # 2. 规则预判：显式多步优化信号
        if user_input and self._hit_complex_marker(user_input):
            return IntentResult(
                name=INTENT_RESUME_OPTIMIZE,
                confidence=0.95,
                source="rule",
                reason="命中多步优化信号词，需要 Master 规划",
                is_complex=True,
            )

        # 3. 三层路由识别单技能意图
        skill, confidence, source, reason = self._route(user_input)

        # 4. 后处理与复杂度修正
        intent = self._post_process(
            skill=skill,
            confidence=confidence,
            source=source,
            reason=reason,
            user_input=user_input,
            has_resume=has_resume or has_file,
            has_jd=has_jd,
        )

        logger.info(
            f"[IntentRecognizer] 意图={intent.name} 置信度={intent.confidence:.2f} "
            f"来源={intent.source} 复杂={intent.is_complex}"
        )
        return intent

    # ---------- 内部方法 ----------

    @staticmethod
    def _hit_complex_marker(user_input: str) -> bool:
        lowered = user_input.lower()
        return any(marker in lowered for marker in _COMPLEX_MARKERS)

    def _route(self, user_input: str) -> tuple:
        """调用三层路由，返回 (skill, confidence, source, reason)；失败时降级"""
        if not user_input or not user_input.strip():
            return INTENT_CHAT_GENERAL, 0.6, "rule", "空输入，默认通用对话"

        router = self.router
        if router is None:
            return INTENT_CHAT_GENERAL, 0.4, "fallback", "路由器不可用，默认通用对话"

        try:
            result = router.route(user_input)
            return result.skill, float(result.confidence), result.source, result.reason
        except Exception as e:
            logger.warning(f"[IntentRecognizer] 三层路由异常，降级: {e}")
            return INTENT_CHAT_GENERAL, 0.4, "fallback", f"路由异常: {e}"

    def _post_process(
        self,
        skill: str,
        confidence: float,
        source: str,
        reason: str,
        user_input: str,
        has_resume: bool,
        has_jd: bool,
    ) -> IntentResult:
        """路由结果后处理：意图归一化 + 复杂度判定"""
        skill = (skill or INTENT_CHAT_GENERAL).strip().lower()

        # 未知技能归一化为通用对话
        known_skills = set(_SIMPLE_SKILL_AGENTS.keys()) | {INTENT_CHAT_GENERAL}
        if skill not in known_skills:
            # LLM 层可能返回 resume-optimize 等别名
            if skill in {INTENT_RESUME_OPTIMIZE, "resume_optimize"}:
                return IntentResult(
                    name=INTENT_RESUME_OPTIMIZE, confidence=confidence,
                    source=source, reason=reason, is_complex=True,
                )
            return IntentResult(
                name=INTENT_CHAT_GENERAL, confidence=min(confidence, 0.6),
                source=source, reason=f"未知技能'{skill}'，归一化为通用对话",
            )

        # 通用对话 + 上下文中已有简历 + 用户提到优化 → 升级为复杂优化任务
        if skill == INTENT_CHAT_GENERAL and has_resume:
            optimize_hints = ["优化", "润色", "改", "提升", "改进"]
            if any(h in user_input for h in optimize_hints):
                return IntentResult(
                    name=INTENT_RESUME_OPTIMIZE,
                    confidence=max(confidence, 0.75),
                    source=source,
                    reason="上下文含简历且用户提出优化诉求",
                    is_complex=True,
                )

        # 纯单技能意图，附带执行器映射
        return IntentResult(
            name=skill,
            confidence=confidence,
            source=source,
            reason=reason,
            params={"agent": _SIMPLE_SKILL_AGENTS.get(skill, "chat")},
            is_complex=False,
        )


def detect_multiple_intents(user_input: str) -> List[str]:
    """检测输入中出现的多个技能信号（用于复杂度辅助判断）"""
    from skillhub.keyword_matcher import KeywordMatcher

    hits: List[str] = []
    matcher = KeywordMatcher()
    for skill_name, keywords in KeywordMatcher.SKILL_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in user_input.lower():
                hits.append(skill_name)
                break
    return hits
