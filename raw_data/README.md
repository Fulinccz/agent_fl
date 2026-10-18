# 招聘数据采集系统

职位(JD)和简历数据分开管理，支持独立数据库配置。

## 目录结构

```
raw_data/
├── config/
│   └── settings.py          # 配置文件
├── sql/
│   ├── job_init.sql         # 职位表结构
│   └── resume_init.sql      # 简历表结构
├── alembic/                 # 数据库迁移版本
├── alembic.ini              # Alembic 配置
├── init_db.py               # 建库/建表初始化脚本
├── enrich_resumes.py        # 简历合成增强（天池数据 + 8 类项目模板）
├── src/
│   ├── logger.py            # 日志模块
│   ├── job/                 # 职位模块
│   │   ├── crawler.py       # 爬虫
│   │   ├── database.py      # 数据库操作
│   │   ├── models.py        # 数据模型
│   │   └── collector.py     # 采集主程序
│   └── resume/              # 简历模块
│       ├── models.py        # 数据模型
│       ├── database.py      # 数据库操作
│       └── importer.py      # CSV导入器（内容哈希稳定 ID，重跑幂等）
├── data/                    # SQLite 数据库 / CSV 数据目录（运行时生成，不入 git）
├── .env                     # 环境变量配置（不入 git）
└── requirements.txt         # 依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

所有配置在 `.env` 文件中：

```bash
# 职位(JD)数据库配置
JOB_DB_TYPE=sqlite                    # sqlite 或 mysql
JOB_SQLITE_PATH=data/job_crawler.db   # SQLite路径
# JOB_DB_HOST=localhost               # MySQL配置
# JOB_DB_PORT=3306
# JOB_DB_USER=root
# JOB_DB_PASSWORD=xxx
# JOB_DB_NAME=job_crawler

# 简历数据库配置（可独立配置）
RESUME_DB_TYPE=sqlite
RESUME_SQLITE_PATH=data/resume.db
# RESUME_DB_HOST=localhost
# RESUME_DB_PORT=3306
# RESUME_DB_USER=root
# RESUME_DB_PASSWORD=xxx
# RESUME_DB_NAME=resume_db

# 采集器配置
BATCH_SIZE=100
MAX_PAGES=10
LOG_LEVEL=INFO
```

## 一、职位(JD)采集

### 1. 初始化表结构
```bash
# MySQL：直接执行 SQL
mysql -u root -p < sql/job_init.sql
# 或使用 Alembic 迁移（job / resume 两库都建）
alembic upgrade head
```

### 2. 命令行采集
```bash
# 基础采集（默认关键词）
python -m src.job.collector Python

# 多关键词 + 城市 + 多页
python -m src.job.collector Python Java Go --city 上海 --pages 3

# 只抓列表，不获取JD（更快）
python -m src.job.collector Python --no-jd
```

> 职位 ID 为原始链接 MD5 前 16 位（智联 / 51job 统一规则），重复采集按 `job_id` 去重，可安全重跑。

### 3. 代码中使用
```python
import asyncio
from src.job.crawler import JobCrawler
from src.job.database import JobDatabase

async def main():
    # 爬取职位
    crawler = JobCrawler()
    jobs = await crawler.search_jobs("Python", city="上海", page=1)
    print(f"找到 {len(jobs)} 个职位")

    # 获取JD
    for job in jobs[:3]:
        jd = await crawler.fetch_jd(job['source_url'], job['source'])
        print(f"JD长度: {len(jd) if jd else 0}")

    # 保存到数据库
    db = JobDatabase()
    db_jobs = [{
        'job_id': j['id'],
        'title': j['title'],
        'company': j.get('company'),
        'salary': j.get('salary'),
        'location': j.get('location'),
        'tags': j.get('tags', []),
        'jd': j.get('jd'),
        'source': j.get('source'),
        'source_url': j.get('source_url'),
    } for j in jobs]
    result = db.save_jobs(db_jobs)
    print(f"保存结果: {result}")

    # 查询
    stats = db.get_stats()
    print(f"统计: {stats}")

asyncio.run(main())
```

### 4. 数据库查询
```python
from src.job.database import JobDatabase

db = JobDatabase()

# 查询所有
jobs = db.get_jobs(limit=10)

# 按关键词搜索
jobs = db.get_jobs(keyword="Python", limit=10)

# 按来源筛选
jobs = db.get_jobs(source="zhaopin", limit=10)
```

## 二、简历数据导入

### 数据来源
天池数据集：https://tianchi.aliyun.com/dataset/201566
- 下载 `Chinese_resume_data.csv`
- 放到 `data/` 目录

### 1. 建表
```bash
mysql -u root -p < sql/resume_init.sql
```

### 2. 导入CSV
```bash
# 命令行导入
python -m src.resume.importer data/Chinese_resume_data.csv

# 指定批量大小与最大批次数
python -m src.resume.importer data/Chinese_resume_data.csv --batch-size 200 --max-batches 50
```

> `resume_id` 由姓名 slug + 整行内容哈希生成（跨进程稳定），同一份 CSV 重复导入不会产生重复简历；`skipped` 即跳过的重复行数。
>
> 如需生成"有项目经历"的增强简历（8 类项目描述模板），先导入再执行：`python enrich_resumes.py`。

### 3. 代码中使用
```python
from src.resume.importer import ResumeImporter
from src.resume.database import ResumeDatabase

# 导入CSV
importer = ResumeImporter()
result = importer.import_from_csv("data/Chinese_resume_data.csv")
print(f"导入结果: {result}")

# 查询数据库
db = ResumeDatabase()

# 统计
stats = db.get_stats()
print(f"总简历数: {stats['total_resumes']}")

# 按岗位查询
resumes = db.get_resumes(target_position="算法工程师", limit=10)

# 按学历查询
resumes = db.get_resumes(degree="硕士", limit=10)
```

## 三、数据库说明

### 职位表 (job_descriptions)
| 字段 | 说明 |
|------|------|
| job_id | 职位唯一ID |
| title | 职位标题 |
| company | 公司名称 |
| salary | 薪资文本 |
| location | 工作地点 |
| tags | 标签列表(JSON) |
| jd | 职位描述 |
| source | 数据来源 |
| source_url | 原始链接 |

### 简历表 (resumes)
| 字段 | 说明 |
|------|------|
| resume_id | 简历稳定编号（姓名 slug + 行内容哈希，唯一） |
| gender | 性别 |
| age | 年龄 |
| target_position | 意向岗位 |
| degree | 学历层次 |
| university_type | 院校类别 |
| work_description | 工作描述 |
| project_description | 项目描述（enrich_resumes.py 合成增强） |
| source | 数据来源（默认 tianchi） |
| created_at / updated_at | 创建 / 更新时间 |

## 四、切换 MySQL

修改 `.env`：
```bash
JOB_DB_TYPE=mysql
JOB_DB_HOST=localhost
JOB_DB_PORT=3306
JOB_DB_USER=root
JOB_DB_PASSWORD=你的密码
JOB_DB_NAME=job_crawler

RESUME_DB_TYPE=mysql
RESUME_DB_HOST=localhost
RESUME_DB_PORT=3306
RESUME_DB_USER=root
RESUME_DB_PASSWORD=你的密码
RESUME_DB_NAME=resume_db
```

然后执行 SQL 初始化：
```bash
mysql -u root -p < sql/job_init.sql
mysql -u root -p < sql/resume_init.sql
```
