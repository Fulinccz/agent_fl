from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from logger import get_logger
from .config import Neo4jConfig

logger = get_logger(__name__)


class Neo4jClient:
    """Neo4j 图数据库客户端封装"""

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None
    ):
        self.uri = uri or Neo4jConfig.URI
        self.username = username or Neo4jConfig.USER
        self.password = password or Neo4jConfig.PASSWORD
        self.database = database or Neo4jConfig.DATABASE
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            logger.info(f"Neo4j driver connected: {self.uri}")
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def run(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行 Cypher 查询并返回结果列表"""
        parameters = parameters or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def run_single(self, query: str, parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """执行 Cypher 查询并返回单条结果"""
        parameters = parameters or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            record = result.single()
            return record.data() if record else None

    def init_schema(self):
        """初始化简历知识图谱的 Schema（约束和索引）"""
        constraints = [
            "CREATE CONSTRAINT candidate_resume_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.resume_id IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE",
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT school_name IF NOT EXISTS FOR (s:School) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT position_name IF NOT EXISTS FOR (p:JobPosition) REQUIRE p.name IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX candidate_name_idx IF NOT EXISTS FOR (c:Candidate) ON (c.name)",
            "CREATE INDEX skill_category_idx IF NOT EXISTS FOR (s:Skill) ON (s.category)",
            "CREATE INDEX project_title_idx IF NOT EXISTS FOR (p:Project) ON (p.title)",
        ]

        for cql in constraints + indexes:
            try:
                self.run(cql)
                logger.info(f"Schema executed: {cql[:60]}...")
            except Exception as e:
                logger.warning(f"Schema execution skipped: {e}")

        logger.info("Neo4j schema initialization completed")

    def clear_all(self):
        """清空所有节点和关系（慎用）"""
        self.run("MATCH (n) DETACH DELETE n")
        logger.warning("All nodes and relationships deleted")

    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        node_counts = self.run("""
            CALL apoc.meta.stats() YIELD labels
            RETURN labels
        """)
        rel_counts = self.run("""
            CALL apoc.meta.stats() YIELD relTypesCount
            RETURN relTypesCount
        """)
        return {
            "node_labels": node_counts[0]["labels"] if node_counts else {},
            "relationship_types": rel_counts[0]["relTypesCount"] if rel_counts else {},
            "connected": True
        }

    def health_check(self) -> bool:
        """检查连接健康状态"""
        try:
            self.run("RETURN 1 AS health")
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
