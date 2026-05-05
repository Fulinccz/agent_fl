# Fulin AI - AI 简历智能优化平台

> **私有化部署的 AI Agent 平台，集成简历解析 - JD 匹配 - 智能优化 - 生成改写全链路能力**

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层 (React)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│                   API Gateway (Go/Gin)                      │
│              路由转发 · 限流熔断 · 安全认证                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│  AI Service  │  │   Data Pipeline │  │   RAG       │
│  (FastAPI)   │  │  (Flink+Redis)  │  │  (FAISS)    │
│              │  │                 │  │             │
│ · LLM 推理   │   │ · ETL 清洗      │  │ · 向量检索   │
│ · Agent 路由 │   │ · 数据去重      │   │ · 语义匹配  │
│ · 简历解析    │  │ · 缓存预热       │  │ · 知识增强   │
└──────────────┘  └─────────────────┘  └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              数据层 (MySQL + Redis + FAISS)                  │
│     结构化数据 · 缓存 · 向量库                                │
└─────────────────────────────────────────────────────────────┘
```

### 核心服务

| 服务 | 技术栈 | 职责 |
|------|--------|------|
| **API Gateway** | Go 1.25 + Gin | 反向代理、负载均衡、限流熔断、CORS |
| **AI Service** | Python 3.12 + FastAPI | LLM 推理、Agent 编排、简历解析 |
| **Data Pipeline** | Java + Flink 2.0 | ETL、数据清洗、Redis 缓存预热 |
| **Frontend** | React 18 + TypeScript | 用户交互界面 |

### 基础设施

| 组件 | 用途 |
|------|------|
| **MySQL 8.0** | 结构化数据持久化（职位、简历） |
| **Redis** | 缓存、分布式锁、去重集合 |
| **FAISS** | 向量检索（简历语义匹配） |
| **Kafka** | 消息队列（预留异步解耦能力） |
| **Docker** | 容器化部署 |

---

## 项目结构

```
FulinAI/
├── backend/
│   ├── api-gateway/              # Go API 网关
│   │   ├── internal/
│   │   │   ├── middleware/       # CORS、限流、安全头
│   │   │   ├── proxy/            # 反向代理
│   │   │   └── router/           # 路由配置
│   │   └── cmd/server/main.go
│   │
│   └── api-service/              # Python AI 核心服务
│       ├── api/                  # API 路由（版本化 /v1）
│       ├── agents/               # Agent 框架
│       │   ├── core/             # 基类与接口
│       │   ├── providers/        # 模型提供者（本地/在线）
│       │   └── langgraph/        # 多技能工作流
│       ├── rag/                  # RAG 检索增强
│       │   ├── text_based/       # 文本向量化
│       │   └── db_based/         # 数据库向量化
│       ├── messaging/            # Kafka 生产者（预留）
│       ├── cache/                # Redis 缓存 + 分布式锁
│       ├── middleware/           # 限流中间件
│       └── main.py               # 服务入口
│
├── raw_data/                     # 数据采集模块
│   ├── src/job/                  # JD 爬虫（并发优化）
│   │   ├── crawler.py            # 智联/51job 并发爬取
│   │   └── collector.py          # 采集调度
│   ├── src/resume/               # 简历导入
│   ├── alembic/                  # 数据库迁移
│   └── data/                     # 数据集
│
├── data/                         # 数据管道
│   └── java-pipeline/            # Flink 实时处理
│       └── src/main/java/com/fulin/
│           ├── FlinkResumeJob.java    # 主作业
│           ├── BatchRedisSink.java    # 批量写入 Redis
│           └── ResumeDataCleaner.java # 数据清洗
│
├── docker-compose.yml            # 全链路容器编排
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.12+
- Go 1.25+
- Java 17+（Flink）
- Docker & Docker Compose
- NVIDIA GPU（推荐，用于模型推理）

### 本地开发

**1. AI Service（Python）**

```bash
cd backend/api-service

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

**2. API Gateway（Go）**

```bash
cd backend/api-gateway
go mod download
go run cmd/server/main.go
```

**3. 数据管道（Java/Flink）**

```bash
cd data/java-pipeline
mvn clean package
mvn exec:java -Dexec.mainClass="com.fulin.FlinkResumeJob"
```

**4. 前端（React）**

```bash
cd frontend
npm install
npm run dev
```

---

## 核心功能

### 1. 简历智能解析

从非结构化文本提取结构化信息：
- 技能栈识别与分级
- 项目经历提取（背景/职责/成果）
- 工作经历时间线分析

### 2. JD 智能匹配

多维度评估简历与职位匹配度：
- 关键词匹配（TF-IDF + 语义相似度）
- 技能 gap 分析
- 经验年限匹配

### 3. AI 优化生成

基于目标职位的个性化改写：
- **句式润色**：专业表达转换
- **量化增强**：添加数据指标
- **STAR 法则**：项目描述结构化
- **风格切换**：专业/简洁/说服/学术

### 4. 流式交互

实时展示思考过程和生成结果：
- Server-Sent Events (SSE) 流式输出
- 支持随时停止生成
- 思考过程可视化

---

## 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件

---

**Made with ❤️ by Fulin**
