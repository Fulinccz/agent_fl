"""
数据库初始化脚本
使用 Python 直接连接 MySQL 创建数据库和表
从 .env 文件读取配置
"""
import os
import pymysql
from pathlib import Path

# 加载 .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def get_db_config():
    """从环境变量获取数据库配置"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'charset': 'utf8mb4'
    }


def read_sql_file(filename: str) -> str:
    """读取 SQL 文件内容"""
    sql_path = Path(__file__).parent / 'sql' / filename
    with open(sql_path, 'r', encoding='utf-8') as f:
        return f.read()


def execute_sql(sql: str, db_name: str = None):
    """执行 SQL 语句"""
    config = get_db_config()
    if db_name:
        config['database'] = db_name

    conn = pymysql.connect(**config)
    try:
        with conn.cursor() as cursor:
            # 分割多条 SQL 语句执行
            for statement in sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt + ';')
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        return False
    finally:
        conn.close()


def init_databases():
    """初始化所有数据库"""
    print("=" * 60)
    print("数据库初始化")
    print("=" * 60)

    # 初始化职位数据库
    print("\n[1/2] 初始化职位数据库...")
    job_sql = read_sql_file('job_init.sql')
    if execute_sql(job_sql):
        print("职位数据库初始化完成")
    else:
        print("职位数据库初始化失败")

    # 初始化简历数据库
    print("\n[2/2] 初始化简历数据库...")
    resume_sql = read_sql_file('resume_init.sql')
    if execute_sql(resume_sql):
        print("简历数据库初始化完成")
    else:
        print("简历数据库初始化失败")

    print("\n" + "=" * 60)
    print("初始化完成")
    print("=" * 60)


if __name__ == '__main__':
    init_databases()
