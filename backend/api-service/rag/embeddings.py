import os
from typing import List
from logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ai", "models", "bge-small-zh"))
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