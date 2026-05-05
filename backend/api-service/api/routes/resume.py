"""
简历优化相关路由
/resume/*

多 Agent 协作简历优化流水线：评分 → 建议 → 润色。
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

router = APIRouter(tags=["简历"])
logger = get_logger(__name__)


@router.post(
    "/optimize/stream",
    summary="流式简历优化",
    description="""
启动多 Agent 协作的简历优化流程，流式返回各阶段结果。

**执行阶段（按顺序）：**
1. `score` - 简历评分（匹配度、完整性、专业度）
2. `suggestions` - 优化建议（具体改进点）
3. `polished` - 润色后的完整简历
4. `complete` - 全部完成

**请求参数：**
- `resume`（必填）：简历原文
- `jd`（可选）：目标职位描述，用于针对性优化
- `position_type`（可选）：职位类型

**响应格式：** `application/json`（每行一个 JSON 对象）
    """,
    responses={
        200: {"description": "流式响应"},
        500: {"description": "优化流程错误"},
    },
)
async def resume_optimize_stream(request: Request, data: ResumeOptimizeRequest):
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
