"""
Agent 生成相关路由
/agent/*

提供 LLM 推理能力，支持本地模型和 OpenAI 兼容接口。
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

router = APIRouter(tags=["Agent"])
logger = get_logger(__name__)

agent_service = AgentService()


@router.post(
    "/generate",
    summary="文本生成",
    description="调用 LLM 生成文本，返回完整结果",
    responses={
        200: {"description": "生成成功"},
        500: {"description": "模型推理错误"},
    },
)
async def agent_generate(request: AgentGenerateRequest):
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


@router.post(
    "/generate/stream",
    summary="流式文本生成",
    description="""
流式调用 LLM，逐 chunk 返回生成结果。

**事件类型：**
- `token` / `content` - 文本片段
- `complete` - 生成完成
- `error` - 错误信息

**响应格式：** `application/json`（每行一个 JSON 对象）
    """,
    responses={
        200: {"description": "流式响应"},
    },
)
async def agent_generate_stream(request: Request, data: AgentGenerateRequest):
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
