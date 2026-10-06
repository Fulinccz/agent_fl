import os
from typing import Optional


class Neo4jConfig:
    """Neo4j 图数据库配置

    配置优先级：环境变量 > 本文件默认值
    如需修改连接信息，请直接编辑本文件或设置环境变量。
    """

    URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USER: str = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD: str = os.getenv("NEO4J_PASSWORD", "fulin123456")
    DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    @classmethod
    def to_dict(cls) -> dict:
        return {
            "uri": cls.URI,
            "username": cls.USER,
            "password": cls.PASSWORD,
            "database": cls.DATABASE,
        }
