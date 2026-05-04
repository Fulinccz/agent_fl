"""
Alembic 迁移环境配置

用法：
    alembic revision --autogenerate -m "create tables"
    alembic upgrade head
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import job_db_config

# 读取 alembic.ini 配置
config = context.config

# 覆盖数据库 URL
config.set_main_option("sqlalchemy.url", job_db_config.connection_string)

# 配置日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 导入模型（用于 autogenerate）
target_metadata = None


def run_migrations_offline() -> None:
    """离线模式（生成 SQL 脚本）"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式（直接执行）"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
