"""
JWT 认证中间件
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from services.config import AppSettings
from logger import get_logger

logger = get_logger(__name__)

security = HTTPBearer(auto_error=False)


def _get_secret() -> str:
    config = AppSettings.load()
    return getattr(config, "jwt_secret", "change-me-in-production")


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建 JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})
    return jwt.encode(to_encode, _get_secret(), algorithm="HS256")


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """解码并验证 JWT Token"""
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=["HS256"])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError as e:
        logger.debug("JWT decode failed: %s", e)
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    FastAPI Dependency：获取当前登录用户

    在需要认证的路由中使用：
        async def endpoint(user=Depends(get_current_user))
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = credentials.credentials
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """可选认证 - 不强制要求登录"""
    if credentials is None:
        return None
    return decode_token(credentials.credentials)


class AuthMiddleware:
    """
    认证中间件 - 可配置白名单路径

    用法（在 main.py 中注册）：
        from middleware.auth import AuthMiddleware
        auth_middleware = AuthMiddleware(
            exempt_paths=["/health", "/ready", "/api/v1/auth/login"]
        )
    """

    def __init__(self, exempt_paths: Optional[list] = None):
        self.exempt_paths = set(exempt_paths or [])
        self.exempt_paths.update([
            "/health",
            "/ready",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ])

    async def __call__(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        if any(request.url.path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Missing or invalid Bearer token"}
            )

        token = auth_header[7:]
        payload = decode_token(token)
        if payload is None:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or expired token"}
            )

        request.state.user = payload
        return await call_next(request)
