from typing import List, Dict, Any, Optional
from logger import get_logger
from rag.embeddings import EmbeddingService, get_embedding_service
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
            self._embedding_service = get_embedding_service()
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
            # vector_store 返回的是余弦距离（0=完全相同，越大越不相似），
            # 这里还原为相似度（越大越相似）再与阈值比较
            distance = r.get("distance", 1.0)
            score = max(0.0, min(1.0, 1.0 - distance))
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

    def get_confident_skill(self, candidates: List[VectorCandidate], threshold: float = 0.85,
                            margin: float = 0.05) -> Optional[str]:
        """判断是否有"足够可信"的技能候选（供 L2 层直接返回、跳过 L3 LLM）。

        旧实现要求"恰好一个候选 ≥ threshold"，导致只要输入同时沾到两个技能
        （简历类输入几乎必然如此）就放弃向量结果、下沉到 L3 本地 LLM（实测 11s+）。
        改为：top1 达到高置信阈值，且相对 top2 有领先优势即可直接采纳。
        """
        if not candidates:
            return None

        top1 = candidates[0]
        if top1.score < threshold:
            return None

        if len(candidates) > 1 and (top1.score - candidates[1].score) < margin:
            return None

        return top1.skill
