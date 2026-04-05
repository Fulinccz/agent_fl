# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routes import router
from logger import get_logger
from services.config import AppSettings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AppSettings.load()
    logger.info("api-service starting, environment=%s", config.env)
    yield
    logger.info("api-service shutting down")


app = FastAPI(title="api-service", version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    import torch

    def set_torch_threads():
        try:
            torch.set_num_threads(torch.get_num_threads())
            torch.set_num_interop_threads(torch.get_num_threads())
        except Exception as e:
            logger.warning(f"PyTorch线程数设置失败: {e}")

    set_torch_threads()
    uvicorn.run(app, host="0.0.0.0", port=8000)
