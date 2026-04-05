

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.registry import get_agent
from agents.utils.prompts import ResumePrompts, TextPrompts, PromptManager
from services.agent_service import AgentService
from services.exceptions import AppError
from rag.retriever import RAGRetriever
from logger import get_logger
import json
import os
from datetime import datetime
import asyncio

router = APIRouter()
logger = get_logger(__name__)

# 初始化 Prompt 管理器
prompt_manager = PromptManager()

# 初始化 RAG 检索器
rag_retriever: RAGRetriever | None = None

def get_rag_retriever() -> RAGRetriever:
    global rag_retriever
    if rag_retriever is None:
        rag_retriever = RAGRetriever()
        rag_retriever.initialize_knowledge_base()
        logger.info("RAG Retriever initialized")
    return rag_retriever

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api-service"}

# 确保上传目录存在
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/agent/stream")
async def agent_stream(request: Request, data: "AgentRequest"):
    import time
    call_time = time.strftime('%H:%M:%S')
    logger.info(f"[{call_time}] === agent_stream 被调用 ===")
    logger.info(f"[{call_time}] query={data.query[:50] if data.query else 'empty'}..., model={data.model}")
    logger.info(f"[{call_time}] deepThinking={getattr(data, 'deepThinking', False)}")
    logger.info(f"[{call_time}] taskType={getattr(data, 'taskType', 'chat')}")
    
    agent = get_agent(provider="local", model=data.model)
    
    # RAG 增强检索
    rag_context = ""
    try:
        retriever = get_rag_retriever()
        logger.info(f"[RAG] 开始检索, 查询: {data.query[:30]}...")
        
        context_results = retriever.retrieve(data.query, top_k=3)
        
        if context_results:
            rag_context_parts = []
            for i, ctx in enumerate(context_results, 1):
                source = ctx.get("metadata", {}).get("source", "知识库")
                content = ctx.get("content", "")
                similarity = ctx.get("similarity", 0)
                rag_context_parts.append(f"[参考资料{i}] (来源:{source}, 相关度:{similarity:.0%})\n{content}")
                logger.info(f"[RAG] 参考资料{i}: 相关度={similarity:.2f}, 来源={source}, 内容长度={len(content)}")
            
            rag_context = "\n\n".join(rag_context_parts)
            logger.info(f"[RAG] ✅ 检索完成, 共{len(context_results)}条相关知识已注入Prompt")
        else:
            logger.info("[RAG] ⚠️ 未检索到相关知识, 使用纯LLM生成")
    except Exception as e:
        logger.warning(f"[RAG] ❌ 检索失败, 使用无上下文模式: {e}")
    
    # 统一多维度输出模式 - 一次性返回所有维度
    if getattr(data, 'deepThinking', False):
        system_prompt = """请用中文回答。

先用结束。
之后按以下3个部分输出，每部分用【标题】开头：

【简历评分】
只给分数（1-100）

【优化建议】
简短列出2-3条

【优化结果】
重写一版优化后的简历内容"""
    else:
        system_prompt = """请用中文回答。按以下3个部分输出，每部分用【标题】开头：

【简历评分】
只给分数（1-100）

【优化建议】
简短列出2-3条

【优化结果】
重写一版优化后的简历内容"""
    
    full_prompt = system_prompt + "\n\n"
    if rag_context:
        full_prompt += f"【参考知识库】\n{rag_context}\n\n"
    full_prompt += data.query
    
    async def event_stream():
        try:
            item_count = 0
            for item in agent.generate_with_thoughts(full_prompt):
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
    deepThinking: bool = False
    taskType: str = "chat"  # chat, parse, optimize, score, polish


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


# ==================== RAG 相关接口 ====================

@router.get("/rag/stats")
async def rag_stats():
    try:
        retriever = get_rag_retriever()
        stats = retriever.get_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"RAG stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/init")
async def rag_init(force_rebuild: bool = False):
    try:
        retriever = get_rag_retriever()
        result = retriever.initialize_knowledge_base(force_rebuild=force_rebuild)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"RAG init error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/add")
async def rag_add_document(content: str = Form(...), source: str = Form(None)):
    try:
        retriever = get_rag_retriever()
        metadata = {"source": source or "manual"} if source else {}
        success = retriever.add_document(content, metadata)
        return {"status": "success" if success else "failed"}
    except Exception as e:
        logger.error(f"RAG add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload")
async def rag_upload_knowledge(file: UploadFile = File(...)):
    try:
        retriever = get_rag_retriever()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        knowledge_dir = os.path.join(os.getcwd(), "data", "resume_knowledge")
        os.makedirs(knowledge_dir, exist_ok=True)
        file_path = os.path.join(knowledge_dir, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        result = retriever.add_file(file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"RAG upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/search")
async def rag_search(query: str = Form(...), top_k: int = 5):
    try:
        retriever = get_rag_retriever()
        results = retriever.retrieve(query, top_k=top_k)
        return {"status": "success", "data": results, "count": len(results)}
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
