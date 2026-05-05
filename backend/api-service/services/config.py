from pydantic_settings import BaseSettings
from typing import Optional


class AppSettings(BaseSettings):
    """应用配置 - 全部支持环境变量覆盖"""

    # 基础服务
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
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

    # JWT
    jwt_secret: str = "change-me-in-production"

    model_config = {"extra": "ignore"}

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
