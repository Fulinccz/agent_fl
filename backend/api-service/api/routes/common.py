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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", example="帮我优化一下简历")
    sessionId: Optional[str] = Field(None, description="会话 ID（不传则新建）")
    model: Optional[str] = Field(None, description="指定模型名称")
    enableRag: bool = Field(True, description="是否启用 RAG 检索增强")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 回复内容")
    sessionId: str = Field(..., description="会话 ID")
    messageCount: int = Field(..., description="当前会话消息数")


class SessionListResponse(BaseModel):
    sessions: list = Field(default=[], description="会话列表")
    total: int = Field(..., description="会话总数")


class AgentGenerateRequest(BaseModel):
    prompt: str = Field(..., description="生成提示词", example="用 Python 写一个快速排序")
    provider: str = Field("local", description="模型提供者：local / openai")
    model: Optional[str] = Field(None, description="模型名称")
    temperature: Optional[float] = Field(0.7, description="生成温度（0-2）")
    max_tokens: Optional[int] = Field(2048, description="最大生成长度")


class ResumeOptimizeRequest(BaseModel):
    resume: str = Field(..., description="简历原文", example="拥有 5 年 Java 开发经验...")
    jd: Optional[str] = Field(None, description="目标职位 JD（可选，用于针对性优化）")
    position_type: Optional[str] = Field(None, description="职位类型：后端 / 前端 / 算法 等")


class SkillExecuteRequest(BaseModel):
    skill_name: str = Field(..., description="技能名称", example="resume_score")
    parameters: Dict[str, Any] = Field(default={}, description="技能参数")


def handle_error(err: Exception, message: str = "Operation failed"):
    logger.error(f"{message}: {err}", exc_info=True)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{message}: {str(err)}"
    )
