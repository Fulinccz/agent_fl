"""
公共依赖和工具函数
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime
from fastapi import HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
_UPLOAD_CHUNK = 1024 * 1024        # 1MB 分块，内存占用与文件大小无关
_MAGIC_PREFIX = 512                # 魔数只看文件头，保留前缀即可


@dataclass
class ReceivedUpload:
    """上传接收结果：成功带 path/size，失败带可直接返回的 error"""
    path: str | None = None
    size: int = 0
    error: JSONResponse | None = None


def cleanup_file(file_path: str | None) -> None:
    """删除上传的临时文件（失败不影响主流程，仅记录）"""
    if not file_path:
        return
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        logger.warning("清理临时上传文件失败 %s：%s", file_path, e)


async def receive_upload(
    request: Request, file: UploadFile, allowed_exts: set[str]
) -> ReceivedUpload:
    """接收上传文件：扩展名白名单 → 大小限制 → 分块落盘 → 魔数校验。

    成功返回 path/size；失败返回 error（临时文件已在内部清理），
    调用方直接把 error 返回给客户端即可。落盘文件用完后由调用方 cleanup_file。
    """
    from services.compliance import check_file_magic
    from services.resume_document import sanitize_filename

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        return ReceivedUpload(error=JSONResponse(
            status_code=400,
            content={"error": f"不支持的文件类型：{ext}，支持：{', '.join(sorted(allowed_exts))}"},
        ))

    # 声明长度超限直接拒绝（multipart 编码会有少量开销，留 1KB 余量）
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_UPLOAD_SIZE + 1024:
                return ReceivedUpload(error=JSONResponse(
                    status_code=413, content={"error": "上传内容过大，最大允许 10MB"}))
        except ValueError:
            return ReceivedUpload(error=JSONResponse(
                status_code=400, content={"error": "非法的 Content-Length 头"}))

    # 文件名消毒：防 ../ 路径穿越与 Windows 非法字符
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{sanitize_filename(file.filename)}")

    # 分块落盘 + 累计硬限：即使客户端不发 Content-Length 也不会被无限读入内存/磁盘
    total_size = 0
    magic_prefix = b""
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    buffer.close()
                    cleanup_file(file_path)
                    return ReceivedUpload(error=JSONResponse(
                        status_code=413, content={"error": "文件过大，最大允许 10MB"}))
                if len(magic_prefix) < _MAGIC_PREFIX:
                    magic_prefix += chunk[:_MAGIC_PREFIX - len(magic_prefix)]
                buffer.write(chunk)
    except Exception:
        cleanup_file(file_path)
        raise

    # 魔数校验：防改扩展名伪装（只需文件头前缀）
    magic = check_file_magic(magic_prefix, file.filename or "")
    if not magic.allowed:
        cleanup_file(file_path)
        return ReceivedUpload(error=JSONResponse(
            status_code=magic.code, content={"error": magic.reason}))

    return ReceivedUpload(path=file_path, size=total_size)


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
