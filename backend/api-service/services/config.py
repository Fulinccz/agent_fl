from pydantic_settings import BaseSettings
from typing import Optional
from .app_config import ENV_FILE


class AppSettings(BaseSettings):
    """基础设施配置（连接凭证/地址端口等敏感与环境差异项）

    从仓库根目录 .env 加载，全部支持进程环境变量覆盖。
    业务行为参数（模型/MAS/RAG/记忆）见 services/app_config.py + config.yaml
    """

    # 基础服务
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    # 模型
    model_provider: str = "local"
    model_name: str = "Qwen3___5-4B"

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_db: str = "job_crawler"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_cache_db: int = 1
    redis_lock_db: int = 2

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"

    # Embedding
    embedding_model: Optional[str] = None

    # SQLite Memory
    memory_db_path: Optional[str] = None

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "fulin123456"

    # JWT
    jwt_secret: str = "change-me-in-production"

    # CORS（逗号分隔的域名列表，生产环境必须配置白名单）
    cors_origins: str = "*"

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @classmethod
    def load(cls):
        return cls()

    @property
    def mysql_dsn(self) -> str:
        """MySQL 连接字符串"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
        )
