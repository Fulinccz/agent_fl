

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.registry import get_agent
from services.agent_service import AgentService
from services.exceptions import AppError
from logger import get_logger
import json
import os
from datetime import datetime
import asyncio

router = APIRouter()
logger = get_logger(__name__)

# 确保上传目录存在
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/agent/stream")
async def agent_stream(request: Request, data: "AgentRequest"):
    import time
    call_time = time.strftime('%H:%M:%S')
    logger.info(f"[{call_time}] === agent_stream 被调用 ===")
    logger.info(f"[{call_time}] query={data.query[:50] if data.query else 'empty'}..., model={data.model}")
    
    agent = get_agent(provider="local", model=data.model)
    
    async def event_stream():
        try:
            item_count = 0
            for item in agent.generate_with_thoughts(data.query):
                item_count += 1
                
                # 检查客户端是否已经断开连接
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected after {item_count} items, stopping generation")
                    agent.stop_generation()
                    break
                
                # 以 JSON 行流式输出，前端易于解析
                yield json.dumps(item, ensure_ascii=False) + "\n"
                
                # 给事件循环一个机会来运行其他任务
                await asyncio.sleep(0)
            
            logger.info(f"[{call_time}] Stream completed, total items: {item_count}")
        except Exception as e:
            logger.error(f"[{call_time}] Stream error: %s", e)
            agent.stop_generation()
        finally:
            logger.info(f"[{call_time}] event_stream finally block, calling stop_generation")
            agent.stop_generation()
    
    logger.info(f"[{call_time}] 返回 StreamingResponse")
    return StreamingResponse(event_stream(), media_type="application/json")


class AgentRequest(BaseModel):
    query: str
    provider: str = "local"
    model: str | None = None


@router.post("/agent")
async def agent_query(request: AgentRequest):
    try:
        logger.debug("Received agent request: %s", request.dict())
        result = AgentService().generate(request.query, provider=request.provider, model=request.model)
        logger.info("Agent result returned, len=%d", len(result) if isinstance(result, str) else 0)
        return {"response": result}
    except AppError as err:
        logger.warning("Agent business error: %s", err, exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"code": err.code, "message": str(err), "context": err.context})
    except Exception as err:
        logger.error("agent_query uncaught error: %s", err, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/agent/upload")
async def upload_file(
    file: UploadFile = File(...),
    query: str = Form(...),
    provider: str = Form("local"),
    model: str | None = Form(None)
):
    try:
        # 保存上传的文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file_path}")
        
        # 调用Agent处理文件和查询
        result = AgentService().generate_with_file(query, file_path, provider=provider, model=model)
        
        return {"response": result, "file_path": file_path}
    except Exception as err:
        logger.error("Upload file error: %s", err, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File upload failed")
