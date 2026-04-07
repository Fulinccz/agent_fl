import os
from typing import List, Dict, Any, Optional
from logger import get_logger

logger = get_logger(__name__)

class RAGRetriever:
    def __init__(
        self,
        collection_name: str = "resume_knowledge",
        persist_dir: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ):
        self.collection_name = collection_name
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.persist_dir = persist_dir or os.path.join(base_dir, "data", "vector_db")
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self._embedding_service = None
        self._vector_store = None
        self._document_processor = None
    
    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from .embeddings import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service
    
    @property
    def vector_store(self):
        if self._vector_store is None:
            from .vector_store import VectorStore
            dim = self.embedding_service.dimension
            self._vector_store = VectorStore(
                collection_name=self.collection_name,
                persist_dir=self.persist_dir,
                embedding_service=self.embedding_service,
                dimension=dim
            )
        return self._vector_store
    
    @property
    def document_processor(self):
        if self._document_processor is None:
            from .document_processor import DocumentProcessor
            self._document_processor = DocumentProcessor()
        return self._document_processor
    
    def initialize_knowledge_base(
        self,
        knowledge_dir: str = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        knowledge_dir = knowledge_dir or os.path.join(base_dir, "data", "resume_knowledge")
        
        if not os.path.exists(knowledge_dir):
            logger.warning(f"Knowledge directory not found: {knowledge_dir}")
            os.makedirs(knowledge_dir, exist_ok=True)
            return {"status": "created", "message": f"Created empty knowledge dir: {knowledge_dir}"}
        
        current_count = self.vector_store.count()
        
        if current_count > 0 and not force_rebuild:
            logger.info(f"Knowledge base already has {current_count} documents")
            return {"status": "exists", "count": current_count}
        
        if force_rebuild and current_count > 0:
            logger.info("Rebuilding knowledge base...")
            self.vector_store.delete_collection()
        
        chunks = self.document_processor.process_directory(knowledge_dir)
        
        if not chunks:
            logger.warning("No documents found in knowledge directory")
            return {"status": "empty", "message": "No documents found"}
        
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        added = self.vector_store.add_documents(documents, metadatas, ids)
        
        result = {
            "status": "success",
            "documents_added": added,
            "total_chunks": len(chunks),
            "collection_name": self.collection_name
        }
        
        logger.info(f"Knowledge base initialized: {result}")
        return result
    
    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        metadata = metadata or {}
        
        count = self.vector_store.count()
        doc_id = f"doc_{count}"
        
        added = self.vector_store.add_documents(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return added > 0
    
    def add_file(self, file_path: str) -> Dict[str, Any]:
        chunks = self.document_processor.process_file(file_path)
        
        if not chunks:
            return {"status": "error", "message": "No content extracted"}
        
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [c.chunk_id for c in chunks]
        
        added = self.vector_store.add_documents(documents, metadatas, ids)
        
        return {
            "status": "success",
            "file": os.path.basename(file_path),
            "chunks_added": added
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
        
        results = self.vector_store.query(query, n_results=top_k, where=filter_metadata)
        
        filtered = []
        for result in results:
            distance = result.get("distance", 1.0)
            similarity = 1 - distance
            
            if similarity >= self.similarity_threshold:
                result["similarity"] = round(similarity, 4)
                filtered.append(result)
        
        logger.info(f"Retrieved {len(filtered)} results for query (threshold={self.similarity_threshold})")
        return filtered
    
    def build_rag_prompt(
        self,
        query: str,
        context_results: List[Dict] = None,
        system_instruction: str = None
    ) -> str:
        context_results = context_results or self.retrieve(query)
        
        system_instruction = system_instruction or """你是一个专业的AI简历智能优化助手。请基于以下参考知识来回答用户的问题。如果参考知识中没有相关信息，请根据你的专业知识回答。必须用中文回答。"""
        
        context_text = ""
        if context_results:
            context_parts = []
            for i, ctx in enumerate(context_results, 1):
                source = ctx.get("metadata", {}).get("source", "知识库")
                content = ctx.get("content", "")
                similarity = ctx.get("similarity", 0)
                context_parts.append(f"[参考资料{i}] (来源:{source}, 相关度:{similarity:.0%})\n{content}")
            
            context_text = "\n\n".join(context_parts)
        
        prompt = f"""{system_instruction}

【参考知识】
{context_text if context_text else "(暂无相关参考知识)"}

【用户问题】
{query}

【回答要求】
请基于以上信息，给出专业、准确的回答。"""
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "total_documents": self.vector_store.count(),
            "persist_dir": self.persist_dir,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }