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
    provider: Optional[str] = Field(None, description="模型提供者：local | deepseek（不传用默认）")
    model: Optional[str] = Field(None, description="指定模型名称（不传用该提供者默认）")
    enableRag: bool = Field(True, description="是否启用 RAG 检索增强")
    ragMode: str = Field("hybrid", description="RAG检索模式: vector | graph | hybrid | merge")
    enableGraph: bool = Field(True, description="是否启用知识图谱检索")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 回复内容")
    sessionId: str = Field(..., description="会话 ID")
    messageCount: int = Field(..., description="当前会话消息数")


class SessionListResponse(BaseModel):
    sessions: list = Field(default=[], description="会话列表")
    total: int = Field(..., description="会话总数")


class ResumeOptimizeRequest(BaseModel):
    resume: str = Field(..., description="简历原文", example="拥有 5 年 Java 开发经验...")
    jd: Optional[str] = Field(None, description="目标职位 JD（可选，用于针对性优化）")
    position_type: Optional[str] = Field(None, description="职位类型：后端 / 前端 / 算法 等")
    provider: Optional[str] = Field(None, description="模型提供者：local | deepseek（不传用默认）")
    model: Optional[str] = Field(None, description="模型名称（不传用该提供者默认）")


class SkillExecuteRequest(BaseModel):
    skill_name: str = Field(..., description="技能名称", example="resume_score")
    parameters: Dict[str, Any] = Field(default={}, description="技能参数")


def handle_error(err: Exception, message: str = "Operation failed"):
    logger.error(f"{message}: {err}", exc_info=True)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{message}: {str(err)}"
    )
