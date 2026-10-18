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


# 默认放行的公开路径（精确匹配，不再用 startswith 豁免整棵路由树）
DEFAULT_EXEMPT_PATHS = frozenset({
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
})


class AuthMiddleware:
    """
    JWT 认证中间件 - 可配置白名单路径与开关

    - AUTH_ENABLED=false（demo 默认）：中间件直接放行，前端无登录入口也能正常使用
    - AUTH_ENABLED=true：除白名单（精确路径）外，全部接口要求 Bearer Token

    用法（在 main.py 中注册）：
        from middleware.auth import AuthMiddleware
        auth_middleware = AuthMiddleware(
            exempt_paths={"/api/auth/login"}
        )
    """

    def __init__(self, exempt_paths: Optional[list] = None, enabled: Optional[bool] = None):
        self.exempt_paths = set(DEFAULT_EXEMPT_PATHS)
        if exempt_paths:
            self.exempt_paths.update(exempt_paths)
        # 启动时读取一次开关；测试可通过 enabled 参数显式覆盖
        self.enabled = AppSettings.load().auth_enabled if enabled is None else enabled

    def _is_exempt(self, path: str) -> bool:
        # Swagger UI 的静态资源（/docs/...）随文档页一并放行
        return path in self.exempt_paths or path.startswith("/docs/")

    async def __call__(self, request: Request, call_next):
        if not self.enabled or request.method == "OPTIONS" or self._is_exempt(request.url.path):
            return await call_next(request)

        from fastapi.responses import JSONResponse

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Missing or invalid Bearer token"}
            )

        token = auth_header[7:]
        payload = decode_token(token)
        if payload is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or expired token"}
            )

        request.state.user = payload
        return await call_next(request)
