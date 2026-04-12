from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from api.routes import router
from logger import get_logger
from services.config import AppSettings

logger = get_logger(__name__)


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
        logger.error(f"Failed to initialize skill system: {e}")
    
    yield
    logger.info("api-service shutting down")


app = FastAPI(title="Fulin AI API Service", version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api-service", "version": "1.0.0"}


if __name__ == "__main__":
    import torch
    import os

    def set_torch_threads():
        try:
            torch.set_num_threads(torch.get_num_threads())
            torch.set_num_interop_threads(torch.get_num_threads())
        except Exception as e:
            logger.warning(f"PyTorch 线程数设置失败：{e}")

    set_torch_threads()
    
    # 从环境变量读取端口，默认 8001
    port = int(os.getenv("PORT", "8001"))
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
