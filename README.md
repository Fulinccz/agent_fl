# Fulin AI - AI 简历智能优化平台

> **私有化部署的 AI Agent 平台，集成简历解析 - JD匹配 - 智能优化 - 生成改写全链路能力**

## 🚀 项目简介

Fulin AI 是一个基于本地开源 LLM（Qwen3.5 系列）的 AI 简历智能优化平台，采用现代化的微服务架构，支持：

- **简历智能解析**：从非结构化文本提取结构化信息
- **JD 匹配分析**：多维度评分和匹配度评估
- **智能优化建议**：基于目标职位的个性化改写
- **文本润色**：多种风格转换（专业/简洁/说服/学术）
- **流式输出**：实时展示思考过程和优化结果
- **停止生成**：支持用户随时中止生成过程

## 🏗️ 技术栈

### 后端服务
| 组件 | 技术栈 | 说明 |
|------|--------|------|
| **AI Service** | Python 3.12 + FastAPI | 核心业务逻辑，LLM 推理 |
| **API Gateway** | Go 1.25 + Gin | 反向代理，负载均衡，限流 |
| **Frontend** | React 18 + TypeScript + Vite | 用户界面 |

### AI 能力
| 技术 | 用途 |
|------|------|
| **本地 LLM** | Qwen3.5 系列（0.8B / 4B / 8B）|
| **Transformers** | 模型推理框架 |
| **LangChain 预留** | 多技能 Agent 架构（开发中）|

## 📁 项目结构

```
FulinAI/
├── backend/
│   ├── api-gateway/          # Go API Gateway
│   │   ├── cmd/server/main.go
│   │   ├── internal/
│   │   │   ├── middleware/    # 中间件层
│   │   │   ├── proxy/        # 代理层
│   │   │   ├── router/       # 路由配置
│   │   │   └── service/      # 服务发现
│   │   └── config/config.yaml
│   │
│   └── api-service/          # Python AI Service
│       ├── agents/           # Agent 框架
│       │   ├── core/         # 核心基类
│       │   ├── providers/    # 模型提供者
│       │   ├── tools/        # 工具系统
│       │   ├── services/     # 业务服务
│       │   └── utils/        # 工具函数
│       ├── api/routes.py     # API 路由
│       └── main.py           # 入口文件
│
├── frontend/                 # React 前端
│   ├── src/
│   │   ├── components/      # UI 组件
│   │   ├── hooks/           # 自定义 Hooks
│   │   ├── services/        # 服务层
│   │   ├── context/         # 状态管理
│   │   └── types/           # 类型定义
│   └── App.tsx              # 主应用
│
├── docker-compose.yml        # Docker 编排
└── README.md                # 本文档
```

## 🛠️ 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd FulinAI

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 访问应用
# Frontend: http://localhost:80
# Gateway: http://localhost:8080
# Python Service: http://localhost:8000
```

### 方式二：本地开发

#### 1️⃣ 后端服务 (Python)

```bash
cd backend/api-service

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 可选：安装加速库（推荐用于 CPU 推理）
pip install git+https://github.com/Dao-AILab/causal-conv1d

# 启动服务
python main.py
```

#### 2️⃣ API Gateway (Go)

```bash
cd backend/api-gateway

# 安装依赖
go mod download

# 启动服务
go run cmd/server/main.go
```

#### 3️⃣ 前端 (React)

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

## ⚙️ 配置说明

### API Gateway 配置 (`backend/api-gateway/config/config.yaml`)

```yaml
server:
  port: "8080"
  mode: "debug"  # debug, release, test

python:
  baseURL: "http://localhost:8000"
  agentPath: "/api/agent"
  agentStreamPath: "/api/agent/stream"

rate_limit:
  enabled: false
  requests_per_second: 100
  burst: 50
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PYTHON_BASEURL` | Python 服务地址 | `http://localhost:8000` |
| `GIN_MODE` | Gin 运行模式 | `debug` |
| `VITE_API_URL` | 前端 API 地址 | `/api` |

## 🔧 本地运行加速库配置（推荐）

为了避免大型模型在 CPU 上推理过慢，建议安装以下加速库：

```bash
cd backend/api-service
pip install git+https://github.com/Dao-AILab/causal-conv1d
```

以上包含：
- `flash-attn`: 高性能 Attention 内核
- `causal-conv1d`: 快速因果卷积实现

启动服务后，日志中不应出现 `fast path is not available`，否则说明加速路径仍不可用。

## 📊 API 文档

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agent/stream` | 流式生成（SSE）|
| POST | `/api/agent` | 同步生成 |
| POST | `/api/agent/upload` | 文件上传处理 |
| GET | `/health` | 健康检查 |

### 流式响应格式

```json
{"type": "thought", "content": "思考内容..."}
{"type": "token", "full_text": "生成的文本"}
{"type": "complete", "full_text": "完整结果"}
```

## 🎯 开发调试小贴士

- ⚠️ 首次运行 `Qwen3___5-4B` 模型体积大，加载时间会较长
- 💡 为了快速响应可在 `agents/providers/local.py` 中设置 `max_new_tokens=32`
- ✅ 请确保本机有可用 GPU + CUDA，或使用轻量模型（如 gpt2）做调试
- 🔄 支持热重载：修改代码后自动重启服务
- 📝 日志输出在控制台，便于调试

## 🏗️ 架构设计亮点

### 分层架构
- **Core 层**: 抽象基类和接口定义
- **Provider 层**: 模型提供者（本地/在线）
- **Tool 层**: 可插拔工具系统
- **Service 层**: 业务逻辑组合
- **API 层**: HTTP 接口和路由

### 设计模式
- **工厂模式**: 统一创建 Provider 和 Tool
- **单例模式**: 全局唯一的 Registry
- **策略模式**: 可切换不同的模型实现
- **装饰器模式**: 自动注册工具

### 扩展性
- ✅ 支持 LangChain 集成（预留接口）
- ✅ 支持多模型切换
- ✅ 支持插件化工具扩展
- ✅ 支持分布式部署

## 📝 开发路线图

### Phase 1: 当前状态 ✅
- [x] 核心框架完成
- [x] Provider 层完成
- [x] Tool 系统完成
- [x] Service 层完成
- [x] 向后兼容保证
- [x] Docker 化部署

### Phase 2: LangChain 集成（进行中）
- [ ] ReAct Agent 实现
- [ ] 多步骤工作流编排
- [ ] 记忆和上下文管理
- [ ] 自我反思机制

### Phase 3: 高级特性（规划中）
- [ ] RAG 知识库检索增强
- [ ] Plugin 插件系统
- [ ] 多租户支持
- [ ] 监控和可观测性
- [ ] A/B 测试平台

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen3) - 本地 LLM 模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架
- [Gin](https://gin-gonic.com/) - Go HTTP 框架
- [React](https://react.dev/) - UI 库
- [LangChain](https://python.langchain.com/) - LLM 应用框架

---

**Made with ❤️ by Fulin Team**
