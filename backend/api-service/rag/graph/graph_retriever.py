import re
from typing import List, Dict, Any, Optional
from logger import get_logger
from .neo4j_client import Neo4jClient

logger = get_logger(__name__)


class GraphRetriever:
    """基于 Neo4j 知识图谱的检索器"""

    def __init__(
        self,
        neo4j_client: Neo4jClient = None,
        top_k: int = 5,
        max_hops: int = 2
    ):
        self.client = neo4j_client or Neo4jClient()
        self.top_k = top_k
        self.max_hops = max_hops

    def _extract_skills_from_query(self, query: str) -> List[str]:
        """从查询中提取技能关键词"""
        common_skills = [
            "Java", "Python", "Go", "C++", "C#", "JavaScript", "TypeScript",
            "Spring", "Spring Boot", "Django", "Flask", "FastAPI",
            "React", "Vue", "Angular", "Node.js",
            "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
            "Kafka", "RabbitMQ", "RocketMQ",
            "Docker", "Kubernetes", "K8s", "Jenkins",
            "Hadoop", "Spark", "Flink", "Hive",
            "PyTorch", "TensorFlow", "LangChain", "LLM", "RAG",
            "Android", "iOS", "Flutter",
            "Linux", "Nginx", "Git"
        ]
        found = []
        query_upper = query.upper()
        for skill in sorted(set(common_skills), key=len, reverse=True):
            pattern = r'\b' + re.escape(skill.upper()) + r'\b'
            if re.search(pattern, query_upper):
                found.append(skill)
        return found

    def _extract_position_from_query(self, query: str) -> Optional[str]:
        """从查询中提取岗位关键词"""
        positions = [
            "Java后端", "后端开发", "前端开发", "全栈", "算法工程师",
            "数据开发", "测试开发", "运维", "DevOps", "AI工程师",
            "产品经理", "项目经理", "技术总监", "架构师"
        ]
        for pos in positions:
            if pos in query:
                return pos
        return None

    def retrieve_by_skills(self, skills: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """按技能检索候选人（多技能匹配）"""
        top_k = top_k or self.top_k
        if not skills:
            return []

        query = """
            MATCH (c:Candidate)-[r:HAS_SKILL]->(s:Skill)
            WHERE s.name IN $skills
            WITH c, collect(DISTINCT s.name) AS matched_skills,
                 count(DISTINCT s) AS skill_count,
                 collect(DISTINCT {name: s.name, category: s.category}) AS skill_details
            ORDER BY skill_count DESC, c.resume_id
            LIMIT $top_k
            RETURN c.resume_id AS resume_id,
                   c.name AS name,
                   c.target_position AS target_position,
                   matched_skills,
                   skill_count,
                   skill_details
        """
        results = self.client.run(query, {"skills": skills, "top_k": top_k})

        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("resume_id", ""),
                "name": r.get("name", ""),
                "target_position": r.get("target_position", ""),
                "content": f"候选人{r.get('name', '')}，意向岗位：{r.get('target_position', '')}，掌握技能：{', '.join(r.get('matched_skills', []))}",
                "metadata": {
                    "source": "graph",
                    "type": "skill_match",
                    "matched_skills": r.get("matched_skills", []),
                    "skill_count": r.get("skill_count", 0),
                    "skill_details": r.get("skill_details", [])
                },
                "graph_score": r.get("skill_count", 0),
                "retriever": "graph"
            })
        return formatted

    def retrieve_by_position_and_skills(
        self,
        position: str,
        skills: List[str],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """按岗位+技能组合检索"""
        top_k = top_k or self.top_k

        if skills:
            query = """
                MATCH (c:Candidate)-[:TARGETS]->(p:JobPosition)
                WHERE p.name CONTAINS $position OR c.target_position CONTAINS $position
                MATCH (c)-[r:HAS_SKILL]->(s:Skill)
                WHERE s.name IN $skills
                WITH c, collect(DISTINCT s.name) AS matched_skills,
                     count(DISTINCT s) AS skill_count
                ORDER BY skill_count DESC
                LIMIT $top_k
                RETURN c.resume_id AS resume_id,
                       c.name AS name,
                       c.target_position AS target_position,
                       c.degree AS degree,
                       c.university_type AS university_type,
                       matched_skills,
                       skill_count
            """
            params = {"position": position, "skills": skills, "top_k": top_k}
        else:
            query = """
                MATCH (c:Candidate)-[:TARGETS]->(p:JobPosition)
                WHERE p.name CONTAINS $position OR c.target_position CONTAINS $position
                RETURN c.resume_id AS resume_id,
                       c.name AS name,
                       c.target_position AS target_position,
                       c.degree AS degree,
                       c.university_type AS university_type
                LIMIT $top_k
            """
            params = {"position": position, "top_k": top_k}

        results = self.client.run(query, params)

        formatted = []
        for r in results:
            skill_info = f"，掌握技能：{', '.join(r.get('matched_skills', []))}" if r.get('matched_skills') else ""
            formatted.append({
                "id": r.get("resume_id", ""),
                "name": r.get("name", ""),
                "target_position": r.get("target_position", ""),
                "content": f"候选人{r.get('name', '')}，意向岗位：{r.get('target_position', '')}{skill_info}",
                "metadata": {
                    "source": "graph",
                    "type": "position_skill_match",
                    "degree": r.get("degree", ""),
                    "university_type": r.get("university_type", ""),
                    "matched_skills": r.get("matched_skills", [])
                },
                "graph_score": r.get("skill_count", 1),
                "retriever": "graph"
            })
        return formatted

    def retrieve_similar_candidates(self, resume_id: str, top_k: int = None) -> List[Dict[str, Any]]:
        """基于技能相似度推荐相似候选人"""
        top_k = top_k or self.top_k

        query = """
            MATCH (c1:Candidate {resume_id: $resume_id})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(c2:Candidate)
            WHERE c1 <> c2
            WITH c2, collect(DISTINCT s.name) AS common_skills, count(DISTINCT s) AS common_count
            OPTIONAL MATCH (c2)-[:HAS_SKILL]->(all_s:Skill)
            WITH c2, common_skills, common_count, count(DISTINCT all_s) AS total_skills
            WITH c2, common_skills, common_count,
                 CASE WHEN total_skills > 0 THEN toFloat(common_count) / total_skills ELSE 0 END AS similarity
            ORDER BY similarity DESC, common_count DESC
            LIMIT $top_k
            RETURN c2.resume_id AS resume_id,
                   c2.name AS name,
                   c2.target_position AS target_position,
                   common_skills,
                   common_count,
                   similarity
        """
        results = self.client.run(query, {"resume_id": resume_id, "top_k": top_k})

        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("resume_id", ""),
                "name": r.get("name", ""),
                "target_position": r.get("target_position", ""),
                "content": f"相似候选人：{r.get('name', '')}，共同技能：{', '.join(r.get('common_skills', []))}",
                "metadata": {
                    "source": "graph",
                    "type": "similar_candidate",
                    "common_skills": r.get("common_skills", []),
                    "common_count": r.get("common_count", 0),
                    "similarity": round(r.get("similarity", 0), 4)
                },
                "graph_score": r.get("common_count", 0) * r.get("similarity", 0),
                "retriever": "graph"
            })
        return formatted

    def retrieve_skill_relationships(self, skill_name: str) -> List[Dict[str, Any]]:
        """检索某个技能关联的候选人分布"""
        query = """
            MATCH (s:Skill {name: $skill_name})<-[:HAS_SKILL]-(c:Candidate)
            OPTIONAL MATCH (c)-[:TARGETS]->(p:JobPosition)
            RETURN c.resume_id AS resume_id,
                   c.name AS name,
                   c.target_position AS target_position,
                   collect(DISTINCT p.name) AS positions
            LIMIT 20
        """
        results = self.client.run(query, {"skill_name": skill_name})
        return results

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """主检索入口：解析Query并路由到合适的检索策略"""
        top_k = top_k or self.top_k
        logger.info(f"[GraphRetriever] Query: '{query}'")

        skills = self._extract_skills_from_query(query)
        position = self._extract_position_from_query(query)

        logger.info(f"[GraphRetriever] Extracted skills: {skills}, position: {position}")

        all_results = []

        if position and skills:
            results = self.retrieve_by_position_and_skills(position, skills, top_k)
            all_results.extend(results)
        elif skills:
            results = self.retrieve_by_skills(skills, top_k)
            all_results.extend(results)
        elif position:
            results = self.retrieve_by_position_and_skills(position, [], top_k)
            all_results.extend(results)

        if not all_results:
            logger.warning(f"[GraphRetriever] No structured match found for query: {query}")
            fuzzy_results = self._fuzzy_search(query, top_k)
            all_results.extend(fuzzy_results)

        logger.info(f"[GraphRetriever] Retrieved {len(all_results)} results")
        return all_results

    def _fuzzy_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """模糊搜索：当结构化匹配失败时回退"""
        cypher = """
            CALL db.index.fulltext.queryNodes('candidateFulltext', $query) YIELD node, score
            RETURN node.resume_id AS resume_id, node.name AS name,
                   node.target_position AS target_position, score
            LIMIT $top_k
        """
        try:
            results = self.client.run(cypher, {"query": query, "top_k": top_k})
            formatted = []
            for r in results:
                formatted.append({
                    "id": r.get("resume_id", ""),
                    "name": r.get("name", ""),
                    "target_position": r.get("target_position", ""),
                    "content": f"候选人{r.get('name', '')}，意向岗位：{r.get('target_position', '')}",
                    "metadata": {"source": "graph", "type": "fuzzy", "score": r.get("score", 0)},
                    "graph_score": r.get("score", 0),
                    "retriever": "graph"
                })
            return formatted
        except Exception as e:
            logger.warning(f"Fuzzy search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取图谱检索器统计"""
        return {
            "type": "graph",
            "top_k": self.top_k,
            "max_hops": self.max_hops,
            "connected": self.client.health_check()
        }
