"""
聊天相关路由
/chat/*

提供多轮长对话能力，支持 RAG 检索增强、会话管理、流式输出。
"""

from __future__ import annotations
import json
import asyncio
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse

from logger import get_logger
from .common import ChatRequest, ChatResponse, SessionListResponse, handle_error
from .deps import get_conversation_graph, get_memory_manager, get_rag_retriever

router = APIRouter(tags=["对话"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=ChatResponse,
    summary="发送消息",
    description="""
发送一条消息到对话系统，返回 AI 回复。

- 不传 `sessionId` 会自动创建新会话
- 默认启用 RAG 检索增强（可通过 `enableRag` 关闭）
- 自动保存对话历史
    """,
    responses={
        200: {"description": "成功", "model": ChatResponse},
        500: {"description": "内部错误"},
    },
)
async def chat(request: ChatRequest):
    try:
        session_id = request.sessionId
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {session_id}")

        graph = get_conversation_graph()

        context = {}
        if request.enableRag:
            try:
                retriever = get_rag_retriever()
                rag_results = retriever.retrieve(request.message, top_k=3)
                if rag_results:
                    rag_context_parts = []
                    for i, ctx in enumerate(rag_results, 1):
                        content = ctx.get("content", "")
                        rag_context_parts.append(f"[参考{i}] {content}")
                    context["rag_context"] = "\n\n".join(rag_context_parts)
                    logger.info(f"[Chat] RAG context added for session {session_id}")
            except Exception as e:
                logger.warning(f"[Chat] RAG failed: {e}")

        response = await graph.chat(
            session_id=session_id,
            user_message=request.message,
            context=context
        )

        mem_manager = get_memory_manager()
        history = await mem_manager.memory_store.get_history(session_id)

        logger.info(f"[Chat] Session {session_id}: {len(history)} messages")

        return ChatResponse(
            response=response,
            sessionId=session_id,
            messageCount=len(history)
        )

    except Exception as err:
        handle_error(err, "Chat failed")


@router.post(
    "/stream",
    summary="流式对话",
    description="""
流式发送消息，通过 SSE 逐 token 返回 AI 回复。

**事件类型：**
- `session` - 会话 ID
- `token` - 文本片段
- `complete` - 完成（含消息数）

**响应格式：** `application/json`（每行一个 JSON 对象）
    """,
    responses={
        200: {"description": "流式响应"},
    },
)
async def chat_stream(request: Request, data: ChatRequest):
    call_time = datetime.now().strftime('%H:%M:%S')
    session_id = data.sessionId or str(uuid.uuid4())
    logger.info(f"[{call_time}] Chat stream started: session={session_id}")

    async def event_stream():
        try:
            yield json.dumps({"type": "session", "sessionId": session_id}, ensure_ascii=False) + "\n"

            graph = get_conversation_graph()

            context = {}
            if data.enableRag:
                try:
                    retriever = get_rag_retriever()
                    rag_results = retriever.retrieve(data.message, top_k=3)
                    if rag_results:
                        rag_context_parts = []
                        for i, ctx in enumerate(rag_results, 1):
                            content = ctx.get("content", "")
                            rag_context_parts.append(f"[参考{i}] {content}")
                        context["rag_context"] = "\n\n".join(rag_context_parts)
                except Exception as e:
                    logger.warning(f"[Chat Stream] RAG failed: {e}")

            full_response = ""
            async for chunk in graph.stream_chat(
                session_id=session_id,
                user_message=data.message,
                context=context
            ):
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected")
                    break

                full_response += chunk
                yield json.dumps({"type": "token", "content": chunk}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            mem_manager = get_memory_manager()
            history = await mem_manager.memory_store.get_history(session_id)
            yield json.dumps({
                "type": "complete",
                "sessionId": session_id,
                "messageCount": len(history)
            }, ensure_ascii=False) + "\n"

            logger.info(f"[{call_time}] Chat stream completed: {len(full_response)} chars")

        except Exception as e:
            logger.error(f"[{call_time}] Chat stream error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="获取会话列表",
    description="获取当前用户的所有对话会话",
)
async def list_sessions(limit: int = 100):
    try:
        sessions = await get_memory_manager().list_sessions(limit)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as err:
        handle_error(err, "List sessions failed")


@router.get(
    "/sessions/{session_id}/history",
    summary="获取会话历史",
    description="获取指定会话的聊天历史记录",
)
async def get_session_history(session_id: str, limit: int = 50):
    try:
        mem_manager = get_memory_manager()
        history = await mem_manager.memory_store.get_history(session_id, limit)
        return {
            "sessionId": session_id,
            "messages": [h.to_dict() for h in history],
            "count": len(history)
        }
    except Exception as err:
        handle_error(err, "Get history failed")


@router.delete(
    "/sessions/{session_id}",
    summary="删除会话",
    description="删除指定会话及其所有历史记录",
)
async def delete_session(session_id: str):
    try:
        await get_memory_manager().delete_session(session_id)
        return {"message": "Session deleted", "sessionId": session_id}
    except Exception as err:
        handle_error(err, "Delete session failed")


@router.delete(
    "/sessions/{session_id}/clear",
    summary="清空会话历史",
    description="清空指定会话的聊天历史，但保留会话本身",
)
async def clear_session_history(session_id: str):
    try:
        await get_memory_manager().clear_session(session_id)
        return {"message": "Session history cleared", "sessionId": session_id}
    except Exception as err:
        handle_error(err, "Clear session failed")
