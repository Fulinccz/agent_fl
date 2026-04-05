import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(
        self,
        collection_name: str = "resume_knowledge",
        persist_dir: str = None,
        embedding_service=None,
        dimension: int = 384
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or os.path.join(os.getcwd(), "data", "vector_db")
        self.embedding_service = embedding_service
        self.dimension = dimension
        
        self._index = None
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []
        self._loaded = False
    
    @property
    def index_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.collection_name}.index")
    
    @property
    def data_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.collection_name}.data")
    
    def _ensure_loaded(self):
        if not self._loaded:
            self._load_or_create()
            self._loaded = True
    
    def _init_index(self):
        self._index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"FAISS index created (dim={self.dimension})")
    
    def _load_or_create(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self._load_from_disk()
        else:
            self._init_index()
    
    def _load_from_disk(self):
        try:
            self._index = faiss.read_index(self.index_path)
            
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                self._documents = data.get("documents", [])
                self._metadatas = data.get("metadatas", [])
                self._ids = data.get("ids", [])
            
            logger.info(f"FAISS loaded from disk: {len(self._documents)} documents")
        except Exception as e:
            logger.warning(f"Failed to load FAISS from disk: {e}, creating new index")
            self._init_index()
    
    def _save_to_disk(self):
        try:
            faiss.write_index(self._index, self.index_path)
            
            with open(self.data_path, 'wb') as f:
                pickle.dump({
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "ids": self._ids
                }, f)
            
            logger.debug(f"FAISS saved to disk: {len(self._documents)} documents")
        except Exception as e:
            logger.error(f"Failed to save FAISS to disk: {e}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List] = None
    ) -> int:
        if not documents:
            return 0
        
        self._ensure_loaded()
        
        if ids is None:
            base_id = len(self._ids)
            ids = [f"doc_{base_id + i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        if embeddings is None and self.embedding_service is not None:
            embeddings = self.embedding_service.encode(documents)
        
        if embeddings is None:
            logger.warning("No embeddings available, cannot add documents")
            return 0
        
        embed_array = np.array(embeddings, dtype=np.float32)
        
        if embed_array.ndim == 1:
            embed_array = embed_array.reshape(1, -1)
        
        faiss.normalize_L2(embed_array)
        
        self._index.add(embed_array)
        
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)
        
        self._save_to_disk()
        
        logger.info(f"Added {len(documents)} documents, total={len(self._documents)}")
        return len(documents)
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Dict = None
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        
        if self._index.ntotal == 0:
            return []
        
        if self.embedding_service is not None:
            query_embedding = np.array([self.embedding_service.encode_single(query_text)], dtype=np.float32)
        else:
            return []
        
        faiss.normalize_L2(query_embedding)
        
        k = min(n_results, self._index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        seen_ids = set()
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            
            doc_id = self._ids[idx]
            if doc_id in seen_ids:
                continue
            
            metadata = self._metadatas[idx] if idx < len(self._metadatas) else {}
            
            if where:
                match = True
                for key, val in where.items():
                    if metadata.get(key) != val:
                        match = False
                        break
                if not match:
                    continue
            
            results.append({
                "content": self._documents[idx],
                "metadata": metadata,
                "distance": float(1 - distances[0][i]),
                "id": doc_id
            })
            seen_ids.add(doc_id)
        
        return results
    
    @property
    def index(self):
        self._ensure_loaded()
        return self._index
    
    def count(self) -> int:
        self._ensure_loaded()
        return self._index.ntotal
    
    def delete_collection(self):
        self._index = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        self._loaded = False
        
        for path in [self.index_path, self.data_path]:
            if os.path.exists(path):
                os.remove(path)
        
        logger.info(f"Collection '{self.collection_name}' deleted")