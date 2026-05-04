import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional

from .redis_client import RAGRedisClient
from .db_vectorizer import DBBasedVectorizer
from logger import get_logger

logger = get_logger(__name__)


class DBBasedRetriever:
    """基于 DB/Redis 的检索器"""

    def __init__(
        self,
        collection_name: str = "resume_db_vectors",
        persist_dir: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._vectorizer = None

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = DBBasedVectorizer(
                collection_name=self.collection_name,
                persist_dir=self.persist_dir
            )
        return self._vectorizer

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None
    ) -> List[Dict[str, Any]]:
        """检索相似简历"""
        top_k = top_k or self.top_k

        results = self.vectorizer.vector_store.query(
            query_text=query,
            n_results=top_k,
            where=filter_metadata
        )

        filtered = []
        for result in results:
            distance = result.get("distance", 1.0)
            similarity = 1 - distance

            if similarity >= self.similarity_threshold:
                result["similarity"] = round(similarity, 4)
                filtered.append(result)

        logger.info(f"DB-based retrieved {len(filtered)} results for query (threshold={self.similarity_threshold})")
        return filtered

    def retrieve_with_resume_details(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None
    ) -> List[Dict[str, Any]]:
        """检索并返回完整简历详情"""
        results = self.retrieve(query, top_k, filter_metadata)

        enriched = []
        for result in results:
            resume_id = result.get("metadata", {}).get("resume_id")
            if resume_id:
                resume = self.vectorizer.redis.get_resume(resume_id)
                if resume:
                    result["resume_detail"] = resume
            enriched.append(result)

        return enriched

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.vectorizer.get_stats()


def test_retrieval(query: str = "Java后端开发"):
    """测试检索"""
    retriever = DBBasedRetriever()
    print(f"\n[TEST] Query: '{query}'\n")

    results = retriever.retrieve(query, top_k=3)

    if not results:
        print("[WARN] No results found")
        return

    for i, result in enumerate(results, 1):
        similarity = result.get("similarity", 0)
        content = result.get("content", "")[:200]
        meta = result.get("metadata", {})
        print(f"\n[Result {i}] Similarity: {similarity:.1%}")
        print(f"  Position: {meta.get('target_position', 'N/A')}")
        print(f"  Degree: {meta.get('degree', 'N/A')}")
        print("-" * 40)
        print(content + "..." if len(result.get("content", "")) > 200 else content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test DB-based retrieval")
    parser.add_argument("--query", type=str, default="Java后端开发", help="查询内容")
    args = parser.parse_args()
    test_retrieval(args.query)
