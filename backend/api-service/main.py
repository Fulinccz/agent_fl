# main.py
from fastapi import FastAPI

from api.routes import router
from logger import get_logger
from services.config import AppSettings

logger = get_logger(__name__)

app = FastAPI(title="api-service", version="0.1.0")
app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    config = AppSettings.load()
    logger.info("api-service starting, environment=%s", config.env)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("api-service shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
