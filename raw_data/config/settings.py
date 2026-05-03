"""
配置文件 - 数据库和 API 配置
"""

import os
from dataclasses import dataclass
from pathlib import Path

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


@dataclass
class DatabaseConfig:
    """数据库配置基类"""
    db_type: str = "mysql"
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"
    sqlite_path: str = "data/default.db"

    @property
    def connection_string(self) -> str:
        if self.db_type == "sqlite":
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.sqlite_path}"
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?charset={self.charset}"


class JobDatabaseConfig(DatabaseConfig):
    """职位(JD)数据库配置"""
    def __init__(self):
        super().__init__(
            db_type=os.getenv("JOB_DB_TYPE", os.getenv("DB_TYPE", "mysql")),
            host=os.getenv("JOB_DB_HOST", os.getenv("DB_HOST", "localhost")),
            port=int(os.getenv("JOB_DB_PORT", os.getenv("DB_PORT", "3306"))),
            user=os.getenv("JOB_DB_USER", os.getenv("DB_USER", "root")),
            password=os.getenv("JOB_DB_PASSWORD", os.getenv("DB_PASSWORD", "")),
            database=os.getenv("JOB_DB_NAME", os.getenv("DB_NAME", "job_crawler")),
            sqlite_path=os.getenv("JOB_SQLITE_PATH", "data/job_crawler.db")
        )


class ResumeDatabaseConfig(DatabaseConfig):
    """简历数据库配置"""
    def __init__(self):
        super().__init__(
            db_type=os.getenv("RESUME_DB_TYPE", os.getenv("DB_TYPE", "mysql")),
            host=os.getenv("RESUME_DB_HOST", os.getenv("DB_HOST", "localhost")),
            port=int(os.getenv("RESUME_DB_PORT", os.getenv("DB_PORT", "3306"))),
            user=os.getenv("RESUME_DB_USER", os.getenv("DB_USER", "root")),
            password=os.getenv("RESUME_DB_PASSWORD", os.getenv("DB_PASSWORD", "")),
            database=os.getenv("RESUME_DB_NAME", os.getenv("DB_NAME", "resume_db")),
            sqlite_path=os.getenv("RESUME_SQLITE_PATH", "data/resume.db")
        )


@dataclass
class CollectorConfig:
    """职位采集器配置"""
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    max_pages: int = int(os.getenv("MAX_PAGES", "10"))
    delay_between_requests: float = float(os.getenv("REQUEST_DELAY", "0.5"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/collector.log")


@dataclass
class ResumeConfig:
    """简历导入配置"""
    batch_size: int = int(os.getenv("RESUME_BATCH_SIZE", "100"))
    max_batches: int = int(os.getenv("RESUME_MAX_BATCHES", "10"))


# 全局配置实例
job_db_config = JobDatabaseConfig()
resume_db_config = ResumeDatabaseConfig()
collector_config = CollectorConfig()
resume_config = ResumeConfig()
