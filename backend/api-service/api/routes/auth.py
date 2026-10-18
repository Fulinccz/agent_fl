"""
认证相关路由
/auth/*

提供 JWT 登录、用户信息获取、Token 刷新功能。
"""

import hmac

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from middleware.auth import create_access_token, decode_token, get_current_user
from services.config import AppSettings
from services.rate_limit import check_rate_limit
from logger import get_logger

router = APIRouter(tags=["认证"])
logger = get_logger(__name__)
security = HTTPBearer()

# 登录端点严格限流（5 次/分钟，配额定义在 services/rate_limit.py），缓解暴力破解
LOGIN_RATE_LIMIT_PATH = "/api/auth/login"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, description="用户名", example="admin")
    password: str = Field(..., min_length=1, max_length=128, description="密码", example="admin")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT 访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(default=86400, description="过期时间（秒）")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录",
    description="使用用户名密码登录，返回 JWT Token。演示账号可通过 ADMIN_USERNAME/ADMIN_PASSWORD 环境变量配置",
    responses={
        200: {"description": "登录成功", "model": TokenResponse},
        401: {"description": "凭证无效"},
        429: {"description": "尝试过于频繁"},
    },
)
async def login(request: Request, req: LoginRequest):
    # 端点级限流（与全局兜底限流独立计数）
    allowed, retry_after, _ = check_rate_limit(request, path=LOGIN_RATE_LIMIT_PATH)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="登录尝试过于频繁，请稍后再试",
            headers={"Retry-After": str(retry_after)},
        )

    settings = AppSettings.load()
    # 常量时间比较，避免用户名/密码比较的时序侧信道
    username_ok = hmac.compare_digest(req.username, settings.admin_username)
    password_ok = hmac.compare_digest(req.password, settings.admin_password)
    if username_ok and password_ok:
        token = create_access_token({"sub": settings.admin_username, "role": "admin"})
        return TokenResponse(access_token=token)

    logger.warning("login failed for username=%r from %s", req.username, request.client.host if request.client else "?")
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get(
    "/me",
    summary="获取当前用户",
    description="根据 JWT Token 返回当前登录用户信息",
    responses={
        200: {"description": "成功"},
        401: {"description": "未登录或 Token 过期"},
    },
)
async def me(user=Depends(get_current_user)):
    return {"user_id": user.get("sub"), "role": user.get("role")}


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="刷新 Token",
    description="使用有效 Token 获取新的访问令牌",
    responses={
        200: {"description": "刷新成功"},
        401: {"description": "原 Token 无效"},
    },
)
async def refresh(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    new_token = create_access_token({
        "sub": payload.get("sub"),
        "role": payload.get("role")
    })
    return TokenResponse(access_token=new_token)
