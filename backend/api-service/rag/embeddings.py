import os
import threading
from typing import List
from logger import get_logger

logger = get_logger(__name__)

_shared_service = None
_shared_lock = threading.Lock()

class EmbeddingService:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        from services.config import AppSettings
        config = AppSettings.load()
        self.model_name = model_name or config.embedding_model or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "ai", "models", "bge-small-zh")
        )
        self.device = device
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded successfully")
        return self._model
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        texts = [t if t else "" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()
    
    def encode_single(self, text: str) -> List[float]:
        result = self.encode([text])
        return result[0] if result else []
    
    @property
    def dimension(self) -> int:
        test_embed = self.encode_single("test")
        return len(test_embed)


def get_embedding_service() -> EmbeddingService:
    """进程级共享嵌入服务（双检锁）。

    EmbeddingService 首次使用会在 .model 内惰性 import sentence-transformers/torch，
    本机实测该 import 冷启动约 46s（模型权重加载本身仅 0.1s）；且每个实例各持一份
    权重。技能路由 L2 与 RAG 检索必须共用同一实例，否则重复付冷启动代价并翻倍内存。
    """
    global _shared_service
    if _shared_service is None:
        with _shared_lock:
            if _shared_service is None:
                _shared_service = EmbeddingService()
    return _shared_service