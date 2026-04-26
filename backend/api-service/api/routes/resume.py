"""
简历优化相关路由
/resume/*
"""

from __future__ import annotations
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from logger import get_logger
from agents.langgraph.resume_agents import get_resume_workflow
from .common import ResumeOptimizeRequest

router = APIRouter()
logger = get_logger(__name__)


@router.post("/optimize/stream")
async def resume_optimize_stream(request: Request, data: ResumeOptimizeRequest):
    """
    多 Agent 简历优化流式接口
    
    流式返回每个 Agent 的执行结果：
    1. type="score" - 评分结果
    2. type="suggestions" - 优化建议
    3. type="polished" - 润色后的简历
    4. type="complete" - 全部完成
    """
    call_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"[{call_time}] Resume optimize stream started")

    async def event_stream():
        try:
            workflow = get_resume_workflow()

            for event in workflow.optimize_stream(
                resume=data.resume,
                jd=data.jd,
                position_type=data.position_type
            ):
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected")
                    break

                yield json.dumps(event, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            logger.info(f"[{call_time}] Resume optimize stream completed")

        except Exception as e:
            logger.error(f"[{call_time}] Resume optimize stream error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
