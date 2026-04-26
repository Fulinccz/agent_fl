"""
公共依赖和工具函数
"""

from __future__ import annotations
import os
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from logger import get_logger

logger = get_logger(__name__)

# 确保上传目录存在
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    """长对话请求模型"""
    message: str
    sessionId: Optional[str] = None
    model: Optional[str] = None
    enableRag: bool = True


class ChatResponse(BaseModel):
    """长对话响应模型"""
    response: str
    sessionId: str
    messageCount: int


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: list
    total: int


class AgentGenerateRequest(BaseModel):
    """Agent 生成请求"""
    prompt: str
    provider: str = "local"
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class ResumeOptimizeRequest(BaseModel):
    """简历优化请求"""
    resume: str = Field(..., description="简历内容")
    jd: Optional[str] = Field(None, description="职位描述")
    position_type: Optional[str] = Field(None, description="职位类型")


class SkillExecuteRequest(BaseModel):
    """技能执行请求"""
    skill_name: str
    parameters: Dict[str, Any] = {}


# ============ 错误处理 ============

def handle_error(err: Exception, message: str = "Operation failed"):
    """统一错误处理"""
    logger.error(f"{message}: {err}", exc_info=True)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{message}: {str(err)}"
    )
