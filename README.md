# Fulin AI - AI 简历智能优化平台

> **私有化部署的 AI Agent 平台，集成简历解析 - JD 匹配 - 智能优化 - 评分改写全链路能力**

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层 (React + TS)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼─────────────────────────────────┐
│                   API Gateway (Go / Gin)                    │
│         反向代理 · IP 令牌桶限流 · 安全头 · CORS              │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│  AI Service  │  │   Data Pipeline │  │   RAG       │
│  (FastAPI)   │  │ (Flink 批处理)  │  │ (FAISS+Neo4j)│
│              │  │                 │  │             │
│ · MAS 编排   │  │ · ETL 清洗      │  │ · 向量检索   │
│ · LLM 推理   │  │ · 数据去重      │  │ · 知识图谱   │
│ · 简历优化   │  │ · 缓存预热      │  │ · 语义匹配   │
│ · 合规校验   │  │                 │  │             │
└──────┬───────┘  └─────────────────┘  └─────────────┘
       │                                       │
       └───────────────────┬───────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│        数据层 (MySQL + Redis + FAISS + Neo4j)               │
│   结构化数据 · 缓存 · 向量库 · 知识图谱                       │
└─────────────────────────────────────────────────────────────┘
```

### 核心服务

| 服务 | 技术栈 | 职责 |
|------|--------|------|
| **API Gateway** | Go 1.25 + Gin | 统一入口反向代理（httputil.ReverseProxy，SSE 即时 flush）、IP 令牌桶限流、安全响应头、CORS |
| **AI Service** | Python 3.12 + FastAPI | MAS 多智能体编排、LLM 推理、简历解析优化、合规校验、可选 JWT 认证 |
| **Data Pipeline** | Java 21 + Flink 2.0 | 有界批处理 ETL（MySQL JDBC → Redis）、数据清洗、缓存预热 |
| **Frontend** | React 19 + TypeScript + Vite | 用户交互界面、流式输出展示 |

### 基础设施

| 组件 | 用途 |
|------|------|
| **MySQL 8.0** | 结构化数据持久化（职位、简历） |
| **Redis 8.4** | 缓存、分布式锁、去重集合 |
| **FAISS** | 向量检索（简历语义匹配，索引文件随仓库构建） |
| **Neo4j** | 知识图谱（可选，`docker compose --profile graph up`，技能共现、场景-技能映射） |
| **Kafka** | 消息队列（预留组件，当前数据管道不经过 Kafka；`--profile kafka` 启动） |
| **Langfuse v4** | LLM 可观测性（自托管，需 MinIO + ClickHouse） |
| **llama-server** | 本地 GGUF 模型推理服务（OpenAI 兼容 HTTP） |
| **Docker** | 容器化部署 |

---

## 本地部署

**环境**：Python 3.12 · Go 1.25 · Node 22 · Docker Desktop（Java 21 仅跑 Flink 管道时需要）

**一键启动（Windows）**：

```powershell
powershell -File .\start.ps1   # MySQL/Redis 容器 + AI Service(:8000) + Gateway(:8080) + Frontend(:5173)，就绪后自动开浏览器
powershell -File .\stop.ps1    # 停止
```

首次先装依赖：`pip install -r backend/api-service/requirements.txt`、`npm install --prefix frontend`。本地模型预热约 30-60s，首条回复慢属正常。

非 Windows 手动启动：`docker compose up -d mysql redis`，再开三个终端分别跑 `python backend/api-service/main.py`、`go run backend/api-gateway/cmd/server/main.go`、`npm --prefix frontend run dev`。

### 需要改的文件

默认开箱即用，按需修改（优先级：代码默认值 < config.yaml < 环境变量）：

| 文件 | 何时改 | 关键项 |
|------|--------|--------|
| 根目录 `.env`（复制自 `.env.example`） | 改凭证 / 开鉴权 / 接云端模型 | `MYSQL_*`、`JWT_SECRET`、`AUTH_ENABLED`、`DEEPSEEK_API_KEY` |
| `backend/api-service/config.yaml` | 换模型 / 开 DeepSeek | `models.local.gguf_path`、`models.deepseek.enabled` |
| `langfuse/.env`（复制自 `langfuse/.env.example`） | 仅启用可观测性时 | 加密密钥 + 初始账号 |

### 访问地址

| 界面 | 地址 |
|------|------|
| 前端主界面 | http://localhost:5173 |
| API 文档（Swagger） | http://localhost:8000/docs |
| Langfuse（可选） | http://localhost:3000 |

### 可选组件

- **Docker 全链路**：`docker compose up -d`（默认 frontend / gateway / python / mysql / redis；`--profile graph` 追加 Neo4j、`--profile data` 追加 Flink、`--profile kafka` 追加 Kafka）。注意：python-service 容器内本地推理不可用（llama-server 仅 Windows x64），容器模式只能走 DeepSeek——演示完整 AI 链路请用一键脚本。
- **Langfuse 可观测性**：`cd langfuse && docker compose up -d`，首次需一次性手动建 MinIO 三个 bucket：

  ```bash
  docker run --rm --network langfuse_default --entrypoint sh minio/mc -c "mc alias set lf http://langfuse-minio:9000 minio_root minio_root_pass_123 && mc mb -p lf/langfuse lf/langfuse-batch lf/langfuse-media"
  ```

- **Flink 批处理**：`cd data/java-pipeline && mvn clean package && java -jar target/resume-batch-pipeline-1.0.0.jar`（MySQL → Redis 有界 ETL，跑完自动退出）。

---

## 项目结构

```
FulinAI/
├── backend/
│   ├── api-gateway/              # Go API 网关
│   │   ├── cmd/server/main.go
│   │   ├── internal/
│   │   │   ├── middleware/       # CORS、限流、安全头
│   │   │   ├── proxy/            # 反向代理 + 流式转发
│   │   │   ├── router/           # 路由配置
│   │   │   └── service/          # 服务发现
│   │   └── pkg/logger/
│   │
│   └── api-service/              # Python AI 核心服务
│       ├── api/routes/           # API 路由（auth, chat, resume, skill, upload）
│       ├── agents/
│       │   ├── core/             # BaseProvider 抽象基类
│       │   ├── mas/              # MAS 多智能体编排
│       │   │   ├── master.py     # 主编排器（意图→规划→执行→反思）
│       │   │   ├── planner.py    # 任务分解与调度
│       │   │   ├── intent.py     # 意图识别
│       │   │   ├── reflector.py  # 反思重规划
│       │   │   └── executors/    # 步骤执行器（chat, polish, score, jd_match, condense）
│       │   ├── providers/
│       │   │   ├── local/        # GGUF + llama-server 本地推理
│       │   │   └── cloud/        # DeepSeek API 云端推理
│       │   └── skills/
│       │       └── resume_agents/  # 简历技能（精简, 匹配, 润色, 评分）
│       ├── rag/                  # RAG 检索增强
│       │   ├── document_processor.py
│       │   ├── embeddings.py
│       │   ├── multimodal.py     # 多模态解析（OCR / 语音转写）
│       │   ├── retriever.py
│       │   ├── vector_store.py
│       │   └── graph/            # Neo4j 知识图谱
│       ├── services/             # 公共服务（合规, 限流, 文档, Tracing, 配置）
│       ├── skillhub/             # 技能路由（关键词→向量→LLM 三层路由）
│       ├── evaluation/           # 评测框架（多 Provider 对比 + LLM-as-Judge）
│       ├── memory/               # 会话记忆（SQLite 持久化）
│       ├── middleware/           # JWT 认证中间件
│       ├── cache/                # Redis 分布式锁
│       ├── scripts/              # 脚本（知识生成, 向量库管理, 评测, 图谱导入）
│       ├── tests/                # 测试套件（469 passed；langfuse 为可选依赖，缺失时 1 个用例自动失败）
│       └── main.py              # 服务入口
│
├── frontend/                     # React + TS 前端
│   └── src/
│       ├── components/           # UI 组件（ErrorBoundary, Chat, Modal）
│       ├── hooks/                # 自定义 Hook（流式响应, 文件上传, 中止控制）
│       ├── services/             # HTTP 客户端 + API 封装
│       └── types/                # TypeScript 类型定义
│
├── raw_data/                     # 数据采集模块
│   ├── src/job/                  # JD 爬虫（智联/51job 并发爬取）
│   ├── src/resume/               # 简历导入（内容哈希稳定 ID，重跑幂等）
│   ├── enrich_resumes.py         # 简历合成增强（天池公开数据 + 8 类项目模板）
│   ├── alembic/                  # 数据库迁移
│   └── data/                     # 数据集
│
├── data/                         # 数据管道
│   └── java-pipeline/            # Flink 有界批处理 ETL（MySQL → Redis）
│
├── ai/models/                    # 模型文件
│   ├── gguf/                     # Qwen3.5-4B GGUF 量化模型
│   ├── bge-small-zh/              # 中文嵌入模型
│   └── whisper/                  # 语音转写模型
│
├── langfuse/                     # Langfuse 自托管（Tracing 可观测性）
│   └── docker-compose.yml
│
├── docker-compose.yml            # 全链路容器编排
└── README.md
```

---

## 核心功能

### 1. MAS 多智能体编排

基于 Master-Slave 架构的任务编排系统：
- **意图识别**：自动区分闲聊 / 简历优化 / 通用复杂任务
- **任务分解**：Planner 按意图生成执行计划（确定性模板 / LLM 分解）
- **并行执行**：评分 + JD 匹配等独立步骤并行批处理
- **反思重规划**：失败步骤自动重试，支持中途中止
- **流式输出**：SSE 实时推送 token、步骤进度、评分结果

### 2. 简历智能解析

从非结构化文本提取结构化信息（支持多模态输入）：
- PDF / Word / 图片（OCR 识别）/ 音频（语音转写）
- 技能栈识别与分级
- 项目经历提取（背景 / 职责 / 成果）
- 工作经历时间线分析

### 3. JD 智能匹配

多维度评估简历与职位匹配度：
- 关键词匹配（TF-IDF + 语义相似度）
- 技能 gap 分析
- 结构化建议（引用简历原文，非模板套话）

### 4. AI 优化生成

基于目标职位的个性化改写：
- **分段 Map-Reduce**：超长简历按段落切分逐段润色，避免截断丢内容
- **句式润色**：专业表达转换（"做" → "主导"）
- **量化增强**：添加数据指标
- **风格切换**：专业 / 简洁 / 说服 / 学术

### 5. 简历评分

自适应档位评估（按简历长度分 simple / medium / complex 三档）：
- 四维度评分：完整性 / 专业度 / 量化程度 / 匹配度
- 维度分析与优化建议（引用简历原文证据）
- 评分与建议合并展示，不再独立卡片

### 6. 简历精简

超长简历（>2800 字）自动触发精简步骤：
- 分段 Map-Reduce 压缩，保留核心信息
- 用户明确要求"不要精简"时自动跳过
- 精简结果作为润色输入，前端按条件显示

### 7. 文档导出

双路径 docx 生成：
- **上传过 .docx**：对齐改写（保留原文件样式、图标，仅替换匹配段落文本）
- **无上传**：模板生成（ResumeDocxSkill 标准排版）
- 导出文件即收即删，不长期堆积

### 8. 合规、限流与认证

- **合规校验**：注入检测 / 手机号 / 身份证（GB11643 校验位）/ 银行卡（Luhn）/ 文件魔数
- **限流**：同一套滑动窗口实现，全局兜底（120 次/分钟）+ 端点级严格配额（流式 10 次/60s、导出 20 次/60s、登录 5 次/60s），按 X-Forwarded-For 第一跳识别客户端，过期桶惰性淘汰；网关层另有每 IP 令牌桶（默认 50 req/s，突发 100）
- 路由层在建立流式连接前拦截，返回 400/403/429；上传按 1MB 分块落盘并硬限 10MB（无 Content-Length 也无法撑爆内存/磁盘）
- **JWT 认证（可选开关）**：`AUTH_ENABLED=false` 时全开放（Demo 默认，前端无登录入口）；置 true 后除精确白名单（健康检查、文档、`/api/auth/login`）外均需 Bearer Token。`ENV=prod` 或开启鉴权但使用默认 `JWT_SECRET` 时服务拒绝启动；演示账号通过 `ADMIN_USERNAME/ADMIN_PASSWORD` 配置

### 9. 评测框架

`evaluation/` 目录提供多 Provider 对比评测：
- 多 Provider 并行跑批（本地 Qwen / DeepSeek）
- LLM-as-Judge 幻觉检测 + 规则评分基线回归
- 裁判熔断全局共享，统计排除失败用例

---

## 测试

```bash
cd backend/api-service
python -m pytest tests/ -q
# 469 passed（未安装可选依赖 langfuse 时，其 1 个专属用例报 ModuleNotFoundError，属环境缺包而非回归）
```

```bash
cd backend/api-gateway && go test ./...     # 网关（CI 上带 -race）
cd frontend && npx tsc --noEmit && npm run lint
cd data/java-pipeline && mvn -q compile    # Flink 管道编译
```

---

## 配置优先级

代码默认值 < `config.yaml` < 环境变量

- **`config.yaml`**：业务行为参数（模型路径、MAS、RAG、Memory 分区）；用 `FULIN_` 前缀 + 双下划线分层的环境变量覆盖（如 `FULIN_RAG__TOP_K=5`）
- **`.env`（仓库根目录）**：基础设施与敏感凭证，变量名无前缀（如 `MYSQL_HOST`、`JWT_SECRET`、`AUTH_ENABLED`、`CORS_ORIGINS`），模板见 `.env.example`
- **Langfuse**：独立使用 `langfuse/.env`（模板 `langfuse/.env.example`），加密密钥不入库

---

## 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件

---

**Made with ❤️ by Fulin**
