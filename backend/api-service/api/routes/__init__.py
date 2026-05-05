"""
API Routes 模块

包含所有 API 路由，按功能模块拆分
"""

from fastapi import APIRouter

from .chat import router as chat_router
from .agent import router as agent_router
from .resume import router as resume_router
from .upload import router as upload_router
from .skill import router as skill_router
from .auth import router as auth_router

router = APIRouter()

router.include_router(auth_router, prefix="/auth")
router.include_router(chat_router, prefix="/chat")
router.include_router(agent_router, prefix="/agent")
router.include_router(resume_router, prefix="/resume")
router.include_router(upload_router)
router.include_router(skill_router, prefix="/skill")


@router.get(
    "/health",
    summary="服务健康检查",
    description="返回服务基本状态，用于 Kubernetes/Docker 存活探针",
    tags=["系统"],
)
async def health_check():
    return {"status": "healthy", "service": "api-service"}
