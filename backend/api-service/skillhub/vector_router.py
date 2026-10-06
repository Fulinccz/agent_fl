from typing import List, Dict, Any, Optional
from logger import get_logger
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore

logger = get_logger(__name__)

PERSIST_DIR = "data/skill_vector_db"
COLLECTION_NAME = "skill_router"


class VectorCandidate:
    def __init__(self, skill: str, score: float, text: str, metadata: Dict[str, Any]):
        self.skill = skill
        self.score = score
        self.text = text
        self.metadata = metadata


class VectorSkillRouter:
    """L2: 向量粗筛层

    将用户输入编码为向量，在技能查询样本库中搜索语义相似的候选技能。
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ):
        self.persist_dir = persist_dir or PERSIST_DIR
        self.collection_name = collection_name or COLLECTION_NAME
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._embedding_service = None
        self._vector_store = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            dim = self.embedding_service.dimension
            self._vector_store = VectorStore(
                collection_name=self.collection_name,
                persist_dir=self.persist_dir,
                embedding_service=self.embedding_service,
                dimension=dim
            )
        return self._vector_store

    def search(self, user_input: str, top_k: int = None) -> List[VectorCandidate]:
        """向量搜索，返回候选技能列表"""
        top_k = top_k or self.top_k

        try:
            results = self.vector_store.query(query_text=user_input, n_results=top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        candidates = []
        for r in results:
            score = r.get("distance", 0)
            if score < self.similarity_threshold:
                continue

            candidates.append(VectorCandidate(
                skill=r["metadata"].get("skill", "unknown"),
                score=score,
                text=r["content"],
                metadata=r["metadata"]
            ))

        # 按 skill 聚合，取最高分的
        skill_best: Dict[str, VectorCandidate] = {}
        for c in candidates:
            if c.skill not in skill_best or c.score > skill_best[c.skill].score:
                skill_best[c.skill] = c

        sorted_candidates = sorted(skill_best.values(), key=lambda x: x.score, reverse=True)
        logger.info(f"Vector router found {len(sorted_candidates)} candidates for: {user_input[:50]}")

        return sorted_candidates

    def get_confident_skill(self, candidates: List[VectorCandidate], threshold: float = 0.85) -> Optional[str]:
        """如果只有一个高置信候选，直接返回"""
        high_conf = [c for c in candidates if c.score >= threshold]
        if len(high_conf) == 1:
            return high_conf[0].skill
        return None
