"""
认证相关路由
/auth/*

提供 JWT 登录、用户信息获取、Token 刷新功能。
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from middleware.auth import create_access_token, decode_token, get_current_user
from logger import get_logger

router = APIRouter(tags=["认证"])
logger = get_logger(__name__)
security = HTTPBearer()


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名", example="admin")
    password: str = Field(..., description="密码", example="admin")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT 访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(default=86400, description="过期时间（秒）")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录",
    description="使用用户名密码登录，返回 JWT Token。演示账号：admin / admin",
    responses={
        200: {"description": "登录成功", "model": TokenResponse},
        401: {"description": "凭证无效"},
    },
)
async def login(req: LoginRequest):
    if req.username == "admin" and req.password == "admin":
        token = create_access_token({"sub": "admin", "role": "admin"})
        return TokenResponse(access_token=token)

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
