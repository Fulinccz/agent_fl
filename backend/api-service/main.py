import time
import uuid
import asyncio
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

# Prometheus 指标（全局定义，避免重复创建）
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"]
)
ACTIVE_REQUESTS = Gauge(
    "http_active_requests",
    "Currently active requests",
    ["method"]
)

from api.routes import router
from logger import get_logger, get_trace_id, set_trace_id, clear_trace_id
from services.config import AppSettings
from middleware.rate_limiter import rate_limiter
from middleware.auth import AuthMiddleware

logger = get_logger(__name__)

# 优雅关闭标志
_is_shutting_down = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AppSettings.load()
    logger.info("api-service starting, environment=%s", config.env)

    # 初始化 skill 系统
    try:
        from skill_creator import init_skills
        init_skills()
        logger.info("Skill system initialized")
    except Exception as e:
        logger.error("Failed to initialize skill system: %s", e)

    # 后台预加载 LLM 模型（避免第一次请求阻塞）
    import threading
    try:
        from agents.langgraph.resume_agents.workflow import preload_model
        thread = threading.Thread(target=preload_model, daemon=True)
        thread.start()
        logger.info("LLM model preloading started in background thread")
    except Exception as e:
        logger.error("Failed to start model preloading: %s", e)

    # 设置信号处理器（优雅关闭）
    def signal_handler(sig, frame):
        global _is_shutting_down
        _is_shutting_down = True
        logger.info("Received shutdown signal, stopping...")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    yield

    # 关闭阶段
    logger.info("api-service shutting down")

    # 等待活跃请求完成（最多 30 秒）
    wait_time = 0
    while wait_time < 30:
        # 检查是否还有活跃请求
        # 实际项目中可以用 asyncio.Task 计数
        await asyncio.sleep(1)
        wait_time += 1

    logger.info("Shutdown complete")


app = FastAPI(
    title="Fulin AI API Service",
    description="""
## Fulin AI - 简历智能优化平台 API

私有化部署的 AI Agent 平台，集成 **简历解析 - JD 匹配 - 智能优化 - 生成改写** 全链路能力。

### 核心功能模块

| 模块 | 说明 |
|------|------|
| **Chat** | 多轮对话 + RAG 检索增强 |
| **Agent** | LLM 推理与流式生成 |
| **Resume** | 多 Agent 协作简历优化（评分→建议→润色） |
| **Upload** | 文件上传 + 简历解析 + 流式处理 |
| **Skill** | 技能执行引擎 |
| **Auth** | JWT 认证 |

### 认证说明

- 大部分接口需要 `Authorization: Bearer <token>` 头部
- 通过 `/api/v1/auth/login` 获取 Token
- 白名单路径：`/health`, `/ready`, `/metrics`, `/docs`, `/auth/login`

### 错误格式

```json
{
  "error": "错误类型",
  "detail": "详细描述",
  "trace_id": "链路追踪ID"
}
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "认证", "description": "JWT 登录、Token 刷新"},
        {"name": "对话", "description": "多轮对话、会话管理、RAG 增强"},
        {"name": "Agent", "description": "LLM 推理、流式生成"},
        {"name": "简历", "description": "多 Agent 协作简历优化"},
        {"name": "上传", "description": "文件上传与流式处理"},
        {"name": "技能", "description": "技能执行与管理"},
        {"name": "系统", "description": "健康检查、就绪检查、Prometheus 指标"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """链路追踪中间件"""
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())[:16]
    set_trace_id(trace_id)

    start_time = time.time()
    ACTIVE_REQUESTS.labels(method=request.method).inc()

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
        extra={
            "trace_id": trace_id,
            "extra": {
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client": request.client.host if request.client else None,
            }
        }
    )

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            "Request completed: %s %s - %d - %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            duration,
            extra={
                "trace_id": trace_id,
                "extra": {
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            }
        )

        response.headers["X-Trace-Id"] = trace_id
        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Request failed: %s %s - %.3fs - %s",
            request.method,
            request.url.path,
            duration,
            str(e),
            extra={
                "trace_id": trace_id,
                "extra": {
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(e),
                }
            },
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "trace_id": trace_id}
        )

    finally:
        ACTIVE_REQUESTS.labels(method=request.method).dec()
        clear_trace_id()


_auth_middleware = AuthMiddleware(exempt_paths=[
    "/health", "/ready", "/metrics",
    "/docs", "/openapi.json", "/redoc",
    "/api/v1/auth", "/api/auth",
    "/api/chat", "/api/agent", "/api/resume", "/api/upload", "/api/skill",
])

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """JWT 认证中间件"""
    return await _auth_middleware(request, call_next)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """限流中间件 - 按 IP 限流"""
    try:
        await rate_limiter.check(request)
    except Exception:
        return JSONResponse(
            status_code=429,
            content={"error": "Too Many Requests", "retry_after": 1}
        )
    return await call_next(request)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """指标收集中间件"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response


app.include_router(router, prefix="/api")


@app.get(
    "/health",
    tags=["系统"],
    summary="服务健康检查",
    description="返回服务基本状态，用于 Kubernetes/Docker 存活探针（Liveness Probe）",
)
async def health_check():
    checks = {
        "service": "api-service",
        "version": "1.0.0",
        "timestamp": time.time(),
    }

    # 如果正在关闭，返回不健康
    if _is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={**checks, "status": "shutting_down"}
        )

    return {**checks, "status": "healthy"}


@app.get(
    "/ready",
    tags=["系统"],
    summary="服务就绪检查",
    description="检查所有依赖服务（MySQL、Redis、Kafka）是否可用，用于 Kubernetes/Docker 就绪探针（Readiness Probe）",
)
async def readiness_check():
    checks = {}

    config = AppSettings.load()

    # 检查 MySQL
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(config.mysql_dsn, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["mysql"] = "ok"
    except Exception as e:
        checks["mysql"] = f"error: {str(e)}"

    # 检查 Redis
    try:
        import redis
        r = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
            socket_connect_timeout=2
        )
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    # 检查 Kafka（可选）
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            retries=1,
            request_timeout_ms=2000
        )
        producer.close()
        checks["kafka"] = "ok"
    except Exception as e:
        checks["kafka"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ready" if all_ok else "not_ready",
            "checks": checks
        }
    )


@app.get(
    "/metrics",
    tags=["系统"],
    summary="Prometheus 指标",
    description="返回 Prometheus 格式的监控指标，包括请求计数、延迟分布、活跃请求数等",
)
async def metrics():
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import torch

    def set_torch_threads():
        try:
            torch.set_num_threads(torch.get_num_threads())
            torch.set_num_interop_threads(torch.get_num_threads())
        except Exception as e:
            logger.warning("PyTorch 线程数设置失败：%s", e)

    set_torch_threads()

    config = AppSettings.load()
    logger.info("Starting server on %s:%d", config.host, config.port)

    uvicorn.run(app, host=config.host, port=config.port)
