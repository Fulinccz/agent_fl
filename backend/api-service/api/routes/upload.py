"""
文件上传相关路由
/upload
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

router = APIRouter()
logger = get_logger(__name__)

agent_service = AgentService()


@router.post("/agent/upload_stream")
async def upload_file_stream(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(...),
    provider: str = Form("local"),
    model: str | None = Form(None)
):
    """上传文件并流式处理"""
    call_time = datetime.now().strftime('%H:%M:%S')
    file_path = None

    try:
        # 保存上传的文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"[{call_time}] File uploaded: {file_path}")

        # 解析简历
        resume_data = parse_resume(file_path)
        skills = resume_data.get("skills", "")
        projects = resume_data.get("projects", "")

        logger.info(f"[{call_time}] 提取技能长度：{len(skills)}, 项目长度：{len(projects)}")

        # 构建提示
        full_prompt = f"""请优化这份简历：

【技能栏】
{skills or '无'}

【项目描述】
{projects or '无'}

【用户补充说明】
{query}

请重点优化技能栏和项目描述，使其更专业、更具吸引力。"""

        # 流式生成
        async def event_stream():
            try:
                agent = agent_service
                result = agent.generate(full_prompt, provider=provider, model=model)

                # 模拟流式输出
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
