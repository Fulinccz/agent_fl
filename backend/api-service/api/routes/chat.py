"""
聊天相关路由
/chat/*

主流程走主从多智能体编排（意图识别 → Master 规划 → 执行器 → ReAct 反思），
编排器异常时自动降级到 LangGraph ConversationGraph，保证可用性。
"""

from __future__ import annotations
import json
import asyncio
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from logger import get_logger
from .common import ChatRequest, ChatResponse, SessionListResponse, handle_error
from .deps import (
    get_conversation_graph,
    get_master_orchestrator,
    get_memory_manager,
    get_rag_retriever,
    get_hybrid_retriever,
)

router = APIRouter(tags=["对话"])
logger = get_logger(__name__)


async def _build_rag_context(message: str, enable_rag: bool, enable_graph: bool, rag_mode: str) -> dict:
    """检索增强上下文构建（chat 两个端点共用）"""
    if not enable_rag:
        return {}
    try:
        from services.app_config import load_app_config
        top_k = load_app_config().rag.top_k

        if enable_graph:
            retriever = get_hybrid_retriever()
            rag_results = retriever.retrieve(message, top_k=top_k, mode=rag_mode)
        else:
            retriever = get_rag_retriever()
            rag_results = retriever.retrieve(message, top_k=top_k)

        if not rag_results:
            return {}

        rag_context_parts = []
        for i, ctx in enumerate(rag_results, 1):
            content = ctx.get("content", "")
            sources = ctx.get("sources", ["vector"])
            source_tag = "+".join(sources)
            rag_context_parts.append(f"[参考{i}|{source_tag}] {content}")
        return {"rag_context": "\n\n".join(rag_context_parts)}
    except Exception as e:
        logger.warning(f"[Chat] RAG failed: {e}")
        return {}


@router.post(
    "",
    response_model=ChatResponse,
    summary="发送消息",
    description="""
发送一条消息到对话系统，返回 AI 回复。

- 不传 `sessionId` 会自动创建新会话
- 默认启用 RAG 检索增强（可通过 `enableRag` 关闭）
- 自动保存对话历史
- 主流程：多智能体编排（意图识别 → 规划 → 执行 → 反思），异常时降级普通对话
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

        context = await _build_rag_context(
            request.message, request.enableRag, request.enableGraph, request.ragMode
        )
        # 模型切换：透传到 MAS 黑板，执行器按此选择 Provider
        if request.provider:
            context["provider"] = request.provider
        if request.model:
            context["model"] = request.model
        context["session_id"] = session_id  # Langfuse 会话分组

        mem_manager = get_memory_manager()
        await mem_manager.add_user_message(session_id, request.message)

        # 主流程：多智能体编排（在线程池中执行同步流水线，避免阻塞事件循环）
        response_text = ""
        try:
            orchestrator = get_master_orchestrator()
            result = await asyncio.to_thread(
                orchestrator.run, request.message, context
            )
            if result.get("success"):
                response_text = result.get("answer") or ""
                logger.info(
                    f"[Chat] MAS done: intent={result.get('intent', {}).get('name') if isinstance(result.get('intent'), dict) else '?'} "
                    f"len={len(response_text)}"
                )
        except Exception as e:
            logger.warning(f"[Chat] MAS pipeline failed, falling back: {e}")

        # 降级回退：LangGraph 对话图（用户消息已在上方保存，跳过图内重复保存）
        if not response_text:
            graph = get_conversation_graph()
            response_text = await graph.chat(
                session_id=session_id,
                user_message=request.message,
                context=context,
                save_user_message=False,
                provider=request.provider,
                model=request.model,
            )

        if response_text:
            await mem_manager.add_assistant_message(session_id, response_text)

        history = await mem_manager.memory_store.get_history(session_id)
        logger.info(f"[Chat] Session {session_id}: {len(history)} messages")

        return ChatResponse(
            response=response_text,
            sessionId=session_id,
            messageCount=len(history)
        )

    except Exception as err:
        handle_error(err, "Chat failed")


@router.post(
    "/stream",
    summary="流式对话",
    description="""
流式发送消息，通过 SSE 返回多智能体执行过程与 AI 回复。

**事件类型：**
- `session` - 会话 ID
- `intent` / `plan` / `step_start` / `step_result` / `reflection` - 多智能体执行过程
- `token` - 回复文本片段
- `score` / `suggestions` / `polished` - 简历优化阶段产物（若触发）
- `complete` - 完成（含消息数）
- `error` - 错误信息

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

            context = await _build_rag_context(
                data.message, data.enableRag, data.enableGraph, data.ragMode
            )
            # 模型切换：透传到 MAS 黑板，执行器按此选择 Provider
            if data.provider:
                context["provider"] = data.provider
            if data.model:
                context["model"] = data.model
            context["session_id"] = session_id  # Langfuse 会话分组

            mem_manager = get_memory_manager()
            await mem_manager.add_user_message(session_id, data.message)

            # 主流程：多智能体编排（线程池中迭代同步生成器，避免阻塞事件循环）
            full_response = ""
            mas_ok = False
            try:
                orchestrator = get_master_orchestrator()
                async for event in iterate_in_threadpool(
                    orchestrator.run_stream(data.message, context)
                ):
                    if await request.is_disconnected():
                        logger.info(f"[{call_time}] Client disconnected")
                        break

                    etype = event.get("type")
                    if etype == "token":
                        full_response += event.get("content", "")
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0)
                mas_ok = True
            except Exception as e:
                logger.warning(f"[{call_time}] MAS stream failed: {e}")

            # 降级回退：LangGraph 对话图（编排器不可用或未产出任何回复时；用户消息已保存，跳过图内保存）
            if not full_response:
                logger.info(f"[{call_time}] Falling back to ConversationGraph")
                graph = get_conversation_graph()
                async for chunk in graph.stream_chat(
                    session_id=session_id,
                    user_message=data.message,
                    context=context,
                    save_user_message=False,
                    provider=data.provider,
                    model=data.model,
                ):
                    if await request.is_disconnected():
                        break
                    token = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                    full_response += token
                    yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0)

            # 保存助手回复
            if full_response:
                await mem_manager.add_assistant_message(session_id, full_response)

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
