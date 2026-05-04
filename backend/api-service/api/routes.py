"""
API Routes - 主路由入口

所有路由已拆分到 routes/ 目录下，按功能模块组织
支持 API 版本控制: /api/v1/xxx
"""

from fastapi import APIRouter

from .routes import router as v1_router

# 创建版本化路由
router = APIRouter()

# v1 版本
router.include_router(v1_router, prefix="/v1")

# 默认指向 v1（兼容旧客户端）
router.include_router(v1_router)

__all__ = ["router"]
