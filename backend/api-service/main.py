import time
import uuid
import asyncio
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

# Prometheus 指标（全局定义，避免重复创建）
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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
ACTIVE_REQUESTS = Counter(
    "http_active_requests",
    "Currently active requests",
    ["method"]
)

from api.routes import router
from logger import get_logger, get_trace_id, set_trace_id, clear_trace_id
from services.config import AppSettings

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


app = FastAPI(title="Fulin AI API Service", version="1.0.0", lifespan=lifespan)


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


@app.get("/health")
async def health_check():
    """健康检查端点（支持 Kubernetes/Docker）"""
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


@app.get("/ready")
async def readiness_check():
    """就绪检查（依赖服务是否可用）"""
    checks = {}

    # 检查 MySQL
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("mysql+pymysql://root:root@localhost:3306/job_crawler", pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["mysql"] = "ok"
    except Exception as e:
        checks["mysql"] = f"error: {str(e)}"

    # 检查 Redis
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    # 检查 Kafka（可选）
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers="localhost:9092",
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


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import torch
    import os

    def set_torch_threads():
        try:
            torch.set_num_threads(torch.get_num_threads())
            torch.set_num_interop_threads(torch.get_num_threads())
        except Exception as e:
            logger.warning("PyTorch 线程数设置失败：%s", e)

    set_torch_threads()

    port = int(os.getenv("PORT", "8001"))
    logger.info("Starting server on port %d", port)

    uvicorn.run(app, host="0.0.0.0", port=port)
