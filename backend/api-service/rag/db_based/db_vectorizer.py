import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any

from .redis_client import RAGRedisClient
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingService
from logger import get_logger

logger = get_logger(__name__)


class DBBasedVectorizer:
    """基于 Redis/DB 的向量化器"""

    def __init__(
        self,
        collection_name: str = "resume_db_vectors",
        persist_dir: str = None,
        redis_client: RAGRedisClient = None
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.redis = redis_client or RAGRedisClient()
        self._embedding_service = None
        self._vector_store = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def vector_store(self):
        if self._vector_store is None:
            dim = self.embedding_service.dimension
            self._vector_store = VectorStore(
                collection_name=self.collection_name,
                persist_dir=self.persist_dir,
                embedding_service=self.embedding_service,
                dimension=dim
            )
        return self._vector_store

    def vectorize_from_redis(
        self,
        limit: int = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """从 Redis 读取简历并向量化的主入口"""
        resumes = self.redis.get_all_resumes(limit=limit)
        if not resumes:
            logger.warning("No resumes found in Redis")
            return {"status": "empty", "message": "No resumes in Redis"}

        logger.info(f"Found {len(resumes)} resumes in Redis, starting vectorization")

        documents = []
        metadatas = []
        ids = []

        for resume in resumes:
            text = self.redis.get_resume_text(resume)
            if not text.strip():
                continue

            rid = resume.get("resume_id", f"resume_{len(ids)}")
            documents.append(text)
            metadatas.append({
                "source": "redis",
                "resume_id": rid,
                "target_position": resume.get("target_position", ""),
                "degree": resume.get("degree", ""),
                "age": resume.get("age", ""),
                "gender": resume.get("gender", ""),
                "university_type": resume.get("university_type", "")
            })
            ids.append(rid)

        if not documents:
            return {"status": "empty", "message": "No valid text extracted"}

        # 分批生成 embedding 并写入
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            embeddings = self.embedding_service.encode(batch_docs)

            added = self.vector_store.add_documents(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
                embeddings=embeddings
            )
            total_added += added
            logger.info(f"Vectorized batch {i // batch_size + 1}: {added} documents")

        return {
            "status": "success",
            "total_resumes": len(resumes),
            "total_vectorized": total_added,
            "collection_name": self.collection_name
        }

    def add_single_resume(self, resume: Dict[str, Any]) -> bool:
        """向量化单条简历"""
        text = self.redis.get_resume_text(resume)
        if not text.strip():
            return False

        rid = resume.get("resume_id", f"resume_{self.vector_store.count()}")
        metadata = {
            "source": "redis",
            "resume_id": rid,
            "target_position": resume.get("target_position", ""),
            "degree": resume.get("degree", ""),
            "age": resume.get("age", ""),
            "gender": resume.get("gender", ""),
            "university_type": resume.get("university_type", "")
        }

        self.vector_store.add_documents(
            documents=[text],
            metadatas=[metadata],
            ids=[rid]
        )
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计"""
        return {
            "collection_name": self.collection_name,
            "vector_count": self.vector_store.count(),
            "redis_stats": self.redis.get_stats()
        }


def run_vectorize(limit: int = None, collection_name: str = "resume_db_vectors"):
    """命令行入口"""
    vectorizer = DBBasedVectorizer(collection_name=collection_name)
    result = vectorizer.vectorize_from_redis(limit=limit)
    print(f"[RESULT] {result}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vectorize resumes from Redis to FAISS")
    parser.add_argument("--limit", type=int, default=None, help="最大处理数量")
    parser.add_argument("--collection", type=str, default="resume_db_vectors", help="Collection名称")
    args = parser.parse_args()
    run_vectorize(limit=args.limit, collection_name=args.collection)
