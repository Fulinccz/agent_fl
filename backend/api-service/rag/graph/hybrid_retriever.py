from typing import List, Dict, Any, Optional
from logger import get_logger
from rag.retriever import RAGRetriever
from .graph_retriever import GraphRetriever
from .neo4j_client import Neo4jClient

logger = get_logger(__name__)


class HybridRetriever:
    """
    混元检索器：融合向量检索 + 图检索，择优返回结果

    策略：
    1. 向量检索负责语义模糊匹配、开放式知识库检索
    2. 图检索负责结构化关系查询、技能关联、多跳推理
    3. 融合层使用 RRF (Reciprocal Rank Fusion) 算法合并排序
    """

    def __init__(
        self,
        vector_retriever: RAGRetriever = None,
        graph_retriever: GraphRetriever = None,
        top_k: int = 5,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
        rrf_k: int = 60,
        enable_vector: bool = True,
        enable_graph: bool = True
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k
        self.enable_vector = enable_vector
        self.enable_graph = enable_graph

        self._neo4j_client = None

    @property
    def neo4j_client(self) -> Neo4jClient:
        if self._neo4j_client is None:
            self._neo4j_client = Neo4jClient()
        return self._neo4j_client

    def _ensure_retrievers(self):
        """延迟初始化检索器"""
        if self.vector_retriever is None and self.enable_vector:
            self.vector_retriever = RAGRetriever()
            self.vector_retriever.initialize_knowledge_base()
            logger.info("Vector retriever auto-initialized")

        if self.graph_retriever is None and self.enable_graph:
            self.graph_retriever = GraphRetriever(neo4j_client=self.neo4j_client)
            logger.info("Graph retriever auto-initialized")

    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion 融合算法

        score = sum(1 / (k + rank)) for each list
        同时考虑向量相似度和图谱关系深度进行加权
        """
        scores: Dict[str, float] = {}
        metadatas: Dict[str, Dict[str, Any]] = {}
        contents: Dict[str, str] = {}
        sources: Dict[str, List[str]] = {}

        for rank, item in enumerate(vector_results):
            doc_id = item.get("id", f"vec_{rank}")
            similarity = item.get("similarity", 0)
            rrf_score = self.vector_weight * (1 / (self.rrf_k + rank + 1))
            rrf_score += self.vector_weight * similarity * 0.3

            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            metadatas[doc_id] = item.get("metadata", {})
            contents[doc_id] = item.get("content", "")
            if doc_id not in sources:
                sources[doc_id] = []
            sources[doc_id].append("vector")

        for rank, item in enumerate(graph_results):
            doc_id = item.get("id", f"graph_{rank}")
            graph_score = item.get("graph_score", 0)
            rrf_score = self.graph_weight * (1 / (self.rrf_k + rank + 1))
            rrf_score += self.graph_weight * min(graph_score * 0.1, 0.3)

            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

            existing_meta = metadatas.get(doc_id, {})
            new_meta = item.get("metadata", {})
            merged_meta = {**existing_meta, **new_meta}
            merged_meta["graph_matched_skills"] = new_meta.get("matched_skills", [])
            merged_meta["graph_type"] = new_meta.get("type", "")
            metadatas[doc_id] = merged_meta

            if not contents.get(doc_id):
                contents[doc_id] = item.get("content", "")
            if doc_id not in sources:
                sources[doc_id] = []
            sources[doc_id].append("graph")

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for doc_id, score in sorted_items[:self.top_k]:
            fused.append({
                "id": doc_id,
                "content": contents.get(doc_id, ""),
                "metadata": metadatas.get(doc_id, {}),
                "fusion_score": round(score, 6),
                "sources": sources.get(doc_id, []),
                "retriever": "hybrid"
            })

        return fused

    def _deduplicate_merge(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        简单去重合并策略：优先保留图谱结果的结构化信息
        """
        seen_ids = set()
        merged = []

        graph_dict = {r.get("id", f"g{i}"): r for i, r in enumerate(graph_results)}

        for v_item in vector_results:
            doc_id = v_item.get("id", "")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            if doc_id in graph_dict:
                g_item = graph_dict[doc_id]
                merged.append({
                    "id": doc_id,
                    "content": v_item.get("content", g_item.get("content", "")),
                    "metadata": {**v_item.get("metadata", {}), **g_item.get("metadata", {})},
                    "similarity": v_item.get("similarity", 0),
                    "graph_score": g_item.get("graph_score", 0),
                    "sources": ["vector", "graph"],
                    "retriever": "hybrid"
                })
            else:
                merged.append({
                    **v_item,
                    "sources": ["vector"],
                    "retriever": "hybrid"
                })

        for g_item in graph_results:
            doc_id = g_item.get("id", "")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            merged.append({
                **g_item,
                "sources": ["graph"],
                "retriever": "hybrid"
            })

        return merged[:self.top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None,
        mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        混元检索主入口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件
            mode: 检索模式 - "vector" | "graph" | "hybrid" | "merge"
        """
        top_k = top_k or self.top_k
        self._ensure_retrievers()

        logger.info(f"[HybridRetriever] mode={mode}, query='{query}'")

        vector_results = []
        graph_results = []

        if mode in ("vector", "hybrid", "merge") and self.enable_vector:
            try:
                vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
                logger.info(f"[HybridRetriever] Vector results: {len(vector_results)}")
            except Exception as e:
                logger.warning(f"[HybridRetriever] Vector retrieval failed: {e}")

        if mode in ("graph", "hybrid", "merge") and self.enable_graph:
            try:
                graph_results = self.graph_retriever.retrieve(query, top_k=top_k * 2)
                logger.info(f"[HybridRetriever] Graph results: {len(graph_results)}")
            except Exception as e:
                logger.warning(f"[HybridRetriever] Graph retrieval failed: {e}")

        if mode == "vector":
            return [{**r, "sources": ["vector"], "retriever": "hybrid"} for r in vector_results[:top_k]]
        elif mode == "graph":
            return [{**r, "sources": ["graph"], "retriever": "hybrid"} for r in graph_results[:top_k]]
        elif mode == "merge":
            return self._deduplicate_merge(vector_results, graph_results)
        else:
            return self._rrf_fusion(vector_results, graph_results)

    def build_rag_prompt(
        self,
        query: str,
        context_results: List[Dict] = None,
        system_instruction: str = None
    ) -> str:
        """构建混元RAG的Prompt"""
        context_results = context_results or self.retrieve(query)

        system_instruction = system_instruction or """你是一个专业的AI简历智能优化助手。请基于以下参考知识来回答用户的问题。
参考知识来自向量语义检索和知识图谱结构检索的融合结果，包含语义相似内容和结构化关联信息。
如果参考知识中没有相关信息，请根据你的专业知识回答。必须用中文回答。"""

        context_text = ""
        if context_results:
            context_parts = []
            for i, ctx in enumerate(context_results, 1):
                source = ctx.get("metadata", {}).get("source", "知识库")
                sources = ctx.get("sources", ["unknown"])
                content = ctx.get("content", "")
                fusion_score = ctx.get("fusion_score", 0)
                similarity = ctx.get("similarity", 0)

                source_tag = "+".join(sources)
                score_info = f"融合分:{fusion_score:.4f}" if fusion_score else f"相似度:{similarity:.0%}"

                extra_info = ""
                matched_skills = ctx.get("metadata", {}).get("graph_matched_skills", [])
                if matched_skills:
                    extra_info = f" [图谱匹配技能: {', '.join(matched_skills)}]"

                context_parts.append(
                    f"[参考{i}] (来源:{source_tag}, {score_info}{extra_info})\n{content}"
                )

            context_text = "\n\n".join(context_parts)

        prompt = f"""{system_instruction}

【参考知识】
{context_text if context_text else "(暂无相关参考知识)"}

【用户问题】
{query}

【回答要求】
请基于以上信息，给出专业、准确的回答。如果参考知识中包含结构化关联信息（如技能匹配），请优先利用这些信息进行推理。"""

        return prompt

    def get_stats(self) -> Dict[str, Any]:
        """获取混元检索器统计"""
        stats = {
            "type": "hybrid",
            "top_k": self.top_k,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "rrf_k": self.rrf_k,
            "enable_vector": self.enable_vector,
            "enable_graph": self.enable_graph
        }

        if self.vector_retriever:
            try:
                stats["vector_stats"] = self.vector_retriever.get_stats()
            except Exception as e:
                stats["vector_stats"] = {"error": str(e)}

        if self.graph_retriever:
            try:
                stats["graph_stats"] = self.graph_retriever.get_stats()
            except Exception as e:
                stats["graph_stats"] = {"error": str(e)}

        return stats
