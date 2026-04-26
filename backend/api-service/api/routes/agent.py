"""
Agent 生成相关路由
/agent/*
"""

from __future__ import annotations
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from logger import get_logger
from services.agent_service import AgentService
from agents.registry import get_agent
from .common import AgentGenerateRequest

router = APIRouter()
logger = get_logger(__name__)

agent_service = AgentService()


@router.post("/generate")
async def agent_generate(request: AgentGenerateRequest):
    """Agent 生成接口"""
    try:
        result = agent_service.generate(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model
        )
        return {"result": result}
    except Exception as e:
        logger.error(f"Agent generate error: {e}")
        return {"error": str(e)}


@router.post("/generate/stream")
async def agent_generate_stream(request: Request, data: AgentGenerateRequest):
    """Agent 流式生成接口"""
    call_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"[{call_time}] Agent generate stream started")

    async def event_stream():
        try:
            agent = get_agent(provider=data.provider, model=data.model)

            for chunk in agent.generate_stream(data.prompt):
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected")
                    break

                yield json.dumps(chunk, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            logger.info(f"[{call_time}] Agent generate stream completed")

        except Exception as e:
            logger.error(f"[{call_time}] Agent generate stream error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
