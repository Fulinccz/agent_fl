from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from .app_config import ENV_FILE

# 占位密钥：仅允许在 AUTH_ENABLED=false 的本地 demo 中出现
DEFAULT_JWT_SECRET = "change-me-in-production"


class AppSettings(BaseSettings):
    """基础设施配置（连接凭证/地址端口等敏感与环境差异项）

    从仓库根目录 .env 加载，全部支持进程环境变量覆盖。
    业务行为参数（模型/MAS/RAG/记忆）见 services/app_config.py + config.yaml
    """

    # 基础服务（alias 锁定环境变量名，避免裸 HOST/PORT 被系统变量意外污染）
    env: str = "dev"
    host: str = Field(default="0.0.0.0", validation_alias="API_SERVICE_HOST")
    port: int = Field(default=8000, validation_alias="API_SERVICE_PORT")
    log_level: str = "INFO"

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

    # Kafka（主链路不依赖；仅 KAFKA_ENABLED=true 时就绪探针才探活）
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_enabled: bool = False

    # Embedding
    embedding_model: Optional[str] = None

    # SQLite Memory
    memory_db_path: Optional[str] = None

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "fulin123456"

    # 鉴权：Demo 默认关闭（前端无登录入口），任何可被访问的环境必须开启
    auth_enabled: bool = False
    jwt_secret: str = DEFAULT_JWT_SECRET

    # 演示账号（生产环境必须通过环境变量覆盖）
    admin_username: str = "admin"
    admin_password: str = "admin"

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
    def cors_origin_list(self) -> list[str]:
        """CORS 白名单；"*" 时返回通配（此时不允许携带凭证）"""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_default_jwt_secret(self) -> bool:
        return self.jwt_secret == DEFAULT_JWT_SECRET

    def validate_security(self) -> None:
        """启动期安全自检，不满足条件直接 fail-fast 拒绝启动"""
        if self.env == "prod" and self.is_default_jwt_secret:
            raise RuntimeError(
                "JWT_SECRET 未配置：ENV=prod 时不允许使用默认占位密钥，"
                "请通过环境变量设置 >=32 位随机串（openssl rand -hex 32）"
            )
        if self.auth_enabled and self.is_default_jwt_secret:
            raise RuntimeError(
                "AUTH_ENABLED=true 时必须配置 JWT_SECRET（>=32 位随机串），"
                "否则签发的 Token 可被任意伪造"
            )

    @property
    def mysql_dsn(self) -> str:
        """MySQL 连接字符串"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
        )
