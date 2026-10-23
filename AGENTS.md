# AGENTS.md — FulinAI 仓库开发约束与指导

> 本文件是仓库级开发的**单一事实来源（SSOT）**。任何 agent / 协作者在本仓库改代码前必须先读本文件。
> 只收录"不知道就会犯错"的项目特有契约；通用最佳实践不重复。

## 项目速览

私有化 AI 简历优化平台：`backend/api-service`（Python 3.12 + FastAPI，MAS 多智能体编排）· `backend/api-gateway`（Go 1.25 + Gin 反向代理）· `frontend`（React 19 + TS + Vite 7）· `data/java-pipeline`（Java 21 + Flink 批处理 ETL）· `raw_data`（爬虫/导入）。本地推理走 GGUF + llama-server 子进程（127.0.0.1:8091，OpenAI 兼容），云端可选 DeepSeek。

## 环境特性（Windows 本机）

- Shell 是 **PowerShell 5**：禁 `&&`/`||`（用 `;` 或 `if ($LASTEXITCODE -eq 0)`）、禁 `cmd /c`、禁 Linux 命令（`ls -la`/`grep -r`/`rm -rf`）
- Python 用仓库根 `.venv`：`.venv\Scripts\python.exe`
- 端口分配：8000 api-service / 8080 gateway / 5173 frontend / 8091 llama-server / 3000 langfuse
- Docker 镜像已预拉取（mysql:8.0、redis:8.4、neo4j:5.26 等），**勿重复 pull**；compose 子网必须是 `172.28.0.0/16`
- 本机无 gcc，Go 测试**不加 `-race`**

## 常用命令

```powershell
powershell -File .\start.ps1                    # 一键启动（MySQL/Redis 容器 + 三服务）
powershell -File .\stop.ps1                     # 停止

cd backend/api-service; python -m pytest tests/ -q   # Python 测试，基线 469 passed
cd backend/api-service; ruff check .            # 只启 E4/E7/E9/F，勿加风格规则
cd backend/api-gateway; go test ./...           # Go 测试
cd frontend; npx tsc --noEmit; npm run lint     # 前端必须 0 error
cd data/java-pipeline; mvn -q compile           # Flink 编译
```

## 硬约束（Do NOT）

- **不主动创建 md 文档 / README / 新文件**，除非用户明确要求；优先编辑既有文件
- **不提交 git**，除非用户明确要求
- CI 门禁必须是真实校验，**禁在测试命令后加 `|| true`**
- 死代码（零引用的旧 hook / 冒烟脚本 / 未调用工具函数）确认后直接删，不留兼容壳
- 派生产物必须入 .gitignore：`data/skill_vector_db/`、`data/resume_knowledge/`、`data/vector_db/`、`.venv/`、`coverage.xml` 等

## 架构契约（改了就会出 bug）

### 配置与凭证

- 优先级：代码默认值 < `config.yaml` < 环境变量。业务参数用 `FULIN_` 前缀 + 双下划线分层（`FULIN_RAG__TOP_K=5`）；基础设施变量无前缀（`MYSQL_HOST`、`JWT_SECRET`）
- 敏感凭证只进根目录 `.env`，业务行为参数只进 `backend/api-service/config.yaml`
- api-service 配置字段用 pydantic `validation_alias` 锁定环境变量名；网关 viper 必须有 `SetEnvKeyReplacer(".", "_")`，否则 compose 注入的 `PYTHON_BASEURL` 被静默忽略

### MAS / LLM 链路

- **两层 error 事件契约勿混用**：executor/provider 内部 chunk 用 `{"type":"error","content":...}`（master 读 `content`）；master 顶层 SSE 事件用 `message`（前端读 `event.message`）
- **Provider 停止标志必须请求级**：GGUF 用 `self._stop_events: set[Event]`；DeepSeek 用 `self._active_streams: list[(Event, holder)]`（**不能用 set——tuple 含 dict 不可哈希**），均配锁，generate_stream 每请求新建 Event、finally 注销
- 意图分类 / 规划 / 路由必须跟随前端传来的 `provider`/`model` 参数，不得固定本地模型
- JSON 输出型 `generate` 必须显式传 `max_new_tokens`（意图分类 256 / JD 匹配 512 / 润色 1024），默认 128 会截断 JSON
- LLM 输出清理复用 `agents/skills/resume_agents/textutil.py` 的 `strip_llm_artifacts`；长文本切分用同文件 `split_text`，**禁止 `resume[:N]` 截断**
- 同步阻塞操作（OCR/语音/LLM 推理）一律 `asyncio.to_thread`
- 本地链路只走 GGUF（transformers 回落已删，registry 失败应抛错而非回落）
- 新增执行器后必须同步 `planner._is_valid_agent` 白名单
- Agent 失败回退必须**写回传入的 state dict 再 return state**（return 新 dict 会被 executor 误判失败触发无谓重试）

### 合规 / 限流 / 上传

- 合规 `check_text` 在建立流式连接前拦截（400/403）；注入词表正则的 alternation 必须用 `(?:...)` 包裹，否则常用词单独成词误拦简历
- 上传/导出必须魔数校验 + `sanitize_filename`；限流只有一套实现（`services/rate_limit.py` 滑动窗口），勿引入第二套
- 全局限流中间件必须 fail-open；`/ready` 含 `engine.dispose()` + 5s 缓存 + MySQL `connect_timeout=2`

### Langfuse 埋点

- **禁用** `update_current_span/update_current_generation`（OTEL contextvars 在 `iterate_in_threadpool` 下跨线程丢失），一律用持有式 `services.tracing.update_observation(obs, output=, usage=)`
- input 截断 500 字符、output 截断 300-500 字符；所有 tracing 调用放 try/except 内（未配置时 no-op）
- gguf 流式 usage：`stream_options.include_usage`，最后一个 chunk `choices=[]` 带 usage，在 finally 回填

### 前端

- `App.tsx` 必须解构渲染 `useStreamResponse` 的 `error`（.error-banner），否则 403/429 无展示
- `useStreamResponse`：ref 与 state 必须同步更新；options 用 `optionsRef` 稳定回调 identity
- 导出 `exportDocx` 带 `AbortSignal.timeout(60_000)`

### 测试与目录纪律

- 测试放 `backend/api-service/tests/`；集成测试（会加载 FAISS/真实 LLM）放 `skillhub/` 避免 pytest 收集
- 直接调 FastAPI endpoint 的 pytest 必须显式传全部 Form 参数，`request.is_disconnected` 用 AsyncMock
- 非核心脚本统一进 `scripts/`；空壳中间层目录上提合并

## 完成前验证清单

改完代码后按影响面执行（全过才算 done）：

1. `pytest tests/ -q`（动了 api-service 必跑）
2. `go test ./...`（动了 gateway 必跑）
3. `npx tsc --noEmit && npm run lint`（动了 frontend 必跑）
4. `ruff check .`（动了任何 Python 文件必跑）
5. 涉及 SSE / provider / 限流的改动：本地 `start.ps1` 起服务实测一遍流式链路
