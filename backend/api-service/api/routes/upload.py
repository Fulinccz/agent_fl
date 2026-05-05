"""
文件上传相关路由
/upload

支持文件上传、简历解析、流式 AI 处理。
"""

from __future__ import annotations
import os
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse

from logger import get_logger
from services.agent_service import AgentService
from rag.document_processor import parse_resume
from .common import UPLOAD_DIR

router = APIRouter(tags=["上传"])
logger = get_logger(__name__)

agent_service = AgentService()


@router.post(
    "/agent/upload_stream",
    summary="上传文件并流式处理",
    description="""
上传文件（如简历 PDF/DOCX），自动解析后调用 AI 流式优化。

**处理流程：**
1. 保存上传的文件到服务器
2. 解析文件内容（提取技能、项目等）
3. 结合用户补充说明构建优化提示词
4. 流式返回 AI 优化结果

**请求格式：** `multipart/form-data`
- `file`：上传的文件（必填）
- `query`：用户补充说明（必填）
- `provider`：模型提供者（可选，默认 local）
- `model`：模型名称（可选）

**响应事件类型：**
- `token` - 文本片段
- `complete` - 完成
- `error` - 错误
    """,
    responses={
        200: {"description": "流式响应"},
        400: {"description": "参数错误或文件解析失败"},
        500: {"description": "内部错误"},
    },
)
async def upload_file_stream(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(...),
    provider: str = Form("local"),
    model: str | None = Form(None)
):
    call_time = datetime.now().strftime('%H:%M:%S')
    file_path = None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"[{call_time}] File uploaded: {file_path}")

        resume_data = parse_resume(file_path)
        skills = resume_data.get("skills", "")
        projects = resume_data.get("projects", "")

        logger.info(f"[{call_time}] 提取技能长度：{len(skills)}, 项目长度：{len(projects)}")

        full_prompt = f"""请优化这份简历：

【技能栏】
{skills or '无'}

【项目描述】
{projects or '无'}

【用户补充说明】
{query}

请重点优化技能栏和项目描述，使其更专业、更具吸引力。"""

        async def event_stream():
            try:
                agent = agent_service
                result = agent.generate(full_prompt, provider=provider, model=model)

                chunk_size = 50
                for i in range(0, len(result), chunk_size):
                    if await request.is_disconnected():
                        break

                    chunk = result[i:i+chunk_size]
                    yield json.dumps({"type": "token", "content": chunk}, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0.05)

                yield json.dumps({"type": "complete"}, ensure_ascii=False) + "\n"

            except Exception as e:
                logger.error(f"[{call_time}] Upload stream error: {e}")
                yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/json")

    except Exception as e:
        logger.error(f"[{call_time}] Upload error: {e}")
        return {"error": str(e)}
