

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from agents.registry import get_agent
from agents.utils.prompts import ResumePrompts, TextPrompts, PromptManager
from agents.langgraph.conversation_graph import ConversationGraph, GraphConfig
from memory.memory_manager import MemoryManager, get_memory_manager
from services.agent_service import AgentService
from services.exceptions import AppError
from rag.retriever import RAGRetriever
from logger import get_logger
import json
import os
from datetime import datetime
import asyncio
import uuid

router = APIRouter()
logger = get_logger(__name__)

# 初始化 Prompt 管理器
prompt_manager = PromptManager()

# 初始化 RAG 检索器
rag_retriever: RAGRetriever | None = None

# 初始化记忆管理器
memory_manager: MemoryManager | None = None

# 初始化 LangGraph 对话图
conversation_graph: ConversationGraph | None = None

def get_rag_retriever() -> RAGRetriever:
    global rag_retriever
    if rag_retriever is None:
        rag_retriever = RAGRetriever()
        rag_retriever.initialize_knowledge_base()
        logger.info("RAG Retriever initialized")
    return rag_retriever


def get_memory_manager() -> MemoryManager:
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager(max_context_length=4000)
        logger.info("Memory Manager initialized")
    return memory_manager


def get_conversation_graph() -> ConversationGraph:
    global conversation_graph
    if conversation_graph is None:
        # 获取本地模型 provider
        agent = get_agent(provider="local", model=None)
        provider = agent.provider if hasattr(agent, 'provider') else agent
        
        config = GraphConfig(
            max_tokens=4000,
            temperature=0.7,
            system_prompt="你是一个专业的 AI 助手，帮助用户优化简历和解答问题。",
            enable_rag=True
        )
        
        conversation_graph = ConversationGraph(
            llm_provider=provider,
            memory_manager=get_memory_manager(),
            config=config
        )
        logger.info("Conversation Graph initialized")
    return conversation_graph

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api-service"}

# 确保上传目录存在
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    """长对话请求模型"""
    message: str
    sessionId: Optional[str] = None  # 不传则创建新会话
    model: Optional[str] = None
    enableRag: bool = True


class ChatResponse(BaseModel):
    """长对话响应模型"""
    response: str
    sessionId: str
    messageCount: int


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: list
    total: int


# ==================== 长对话 API（LangGraph + Memory）====================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    长对话接口 - 支持多轮对话和记忆
    
    - 如果不传 sessionId，会自动创建新会话
    - 支持 RAG 检索增强
    - 自动保存对话历史
    """
    try:
        # 获取或创建会话 ID
        session_id = request.sessionId
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {session_id}")
        
        # 获取对话图
        graph = get_conversation_graph()
        
        # RAG 检索（如果启用）
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
        
        # 执行对话
        response = await graph.chat(
            session_id=session_id,
            user_message=request.message,
            context=context
        )
        
        # 获取对话统计
        mem_manager = get_memory_manager()
        history = await mem_manager.memory_store.get_history(session_id)
        
        logger.info(f"[Chat] Session {session_id}: {len(history)} messages")
        
        return ChatResponse(
            response=response,
            sessionId=session_id,
            messageCount=len(history)
        )
        
    except Exception as err:
        logger.error(f"Chat error: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(err)}"
        )


@router.post("/chat/stream")
async def chat_stream(request: Request, data: ChatRequest):
    """
    长对话流式接口
    
    与 /chat 相同，但使用 Server-Sent Events 流式返回
    """
    import time
    call_time = time.strftime('%H:%M:%S')
    
    # 获取或创建会话 ID
    session_id = data.sessionId or str(uuid.uuid4())
    logger.info(f"[{call_time}] Chat stream started: session={session_id}")
    
    async def event_stream():
        try:
            # 发送会话信息
            yield json.dumps({"type": "session", "sessionId": session_id}, ensure_ascii=False) + "\n"
            
            # 获取对话图
            graph = get_conversation_graph()
            
            # RAG 检索
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
            
            # 执行流式对话
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
            
            # 发送完成标记
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


@router.get("/chat/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = 100):
    """获取会话列表"""
    try:
        sessions = await get_memory_manager().list_sessions(limit)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as err:
        logger.error(f"List sessions error: {err}")
        raise HTTPException(status_code=500, detail=str(err))


@router.get("/chat/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    """获取指定会话的历史记录"""
    try:
        mem_manager = get_memory_manager()
        history = await mem_manager.memory_store.get_history(session_id, limit)
        return {
            "sessionId": session_id,
            "messages": [h.to_dict() for h in history],
            "count": len(history)
        }
    except Exception as err:
        logger.error(f"Get history error: {err}")
        raise HTTPException(status_code=500, detail=str(err))


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        await get_memory_manager().delete_session(session_id)
        return {"message": "Session deleted", "sessionId": session_id}
    except Exception as err:
        logger.error(f"Delete session error: {err}")
        raise HTTPException(status_code=500, detail=str(err))


@router.delete("/chat/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """清空会话历史（保留会话）"""
    try:
        await get_memory_manager().clear_session(session_id)
        return {"message": "Session history cleared", "sessionId": session_id}
    except Exception as err:
        logger.error(f"Clear session error: {err}")
        raise HTTPException(status_code=500, detail=str(err))


@router.post("/agent/upload_stream")
async def upload_file_stream(
    file: UploadFile = File(...),
    query: str = Form(...),
    provider: str = Form("local"),
    model: str | None = Form(None)
):
    import time
    call_time = time.strftime('%H:%M:%S')
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
        
        # 读取文件内容（支持 docx）
        import mammoth
        with open(file_path, 'rb') as f:
            result = mammoth.extract_raw_text(f)
            resume_content = result.value
        
        # 解析简历内容，提取技能和项目
        skills = ""
        projects = ""
        
        # 简单的内容提取逻辑
        lines = resume_content.split('\n')
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            if '技能' in line_stripped and ('：' in line_stripped or ':' in line_stripped):
                current_section = 'skills'
                continue
            elif ('项目' in line_stripped or '经历' in line_stripped) and ('：' in line_stripped or ':' in line_stripped):
                current_section = 'projects'
                continue
            elif line_stripped and ('：' in line_stripped or ':' in line_stripped) and len(line_stripped) < 20:
                current_section = None
                continue
            
            if current_section == 'skills' and line_stripped:
                skills += line + '\n'
            elif current_section == 'projects' and line_stripped:
                projects += line + '\n'
        
        logger.info(f"[{call_time}] 提取技能长度：{len(skills)}, 项目长度：{len(projects)}")
        
        # 默认触发简历润色 skill
        logger.debug(f"[{call_time}] 触发简历润色 skill...")
        try:
            from skill_creator.resume_polishing.resume_polishing import ResumePolishingSkill
            
            # 只处理技能栏和项目描述（提前精简数据）
            content_to_polish = f"""【技能】
{skills if skills else '无'}

【项目经历】
{projects if projects else '无'}"""
            
            # 创建 skill 实例
            skill = ResumePolishingSkill()
            
            # 流式输出 - 分段处理并实时反馈进度
            async def event_stream():
                try:
                    # 发送开始消息到思考过程
                    yield json.dumps({"type": "thought", "content": "开始简历优化...\n\n"}, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0.1)
                    
                    # 解析内容
                    skills_content, projects_content = skill._parse_content(content_to_polish)
                    
                    # 检查是否提取到内容
                    if skills_content == '无' and projects_content == '无':
                        yield json.dumps({"type": "thought", "content": "⚠️ 未能从简历中提取到技能和项目信息\n"}, ensure_ascii=False) + "\n"
                        yield json.dumps({"type": "thought", "content": "请确保简历中包含'技能'、'项目经历'等明确的章节标题\n\n"}, ensure_ascii=False) + "\n"
                    
                    # 处理技能
                    polished_skills_lines = []
                    if skills_content and skills_content != '无':
                        yield json.dumps({"type": "thought", "content": "【正在优化技能描述】\n"}, ensure_ascii=False) + "\n"
                        await asyncio.sleep(0.1)
                        
                        skill_lines = [line.strip() for line in skills_content.split('\n') if line.strip()]
                        total_skills = len(skill_lines)
                        
                        for idx, line in enumerate(skill_lines):
                            if not line or line in ['【技能】', '技能：']:
                                continue
                            
                            # 发送进度到思考过程
                            yield json.dumps({"type": "thought", "content": f"  处理第 {idx+1}/{total_skills} 条技能...\n"}, ensure_ascii=False) + "\n"
                            await asyncio.sleep(0.05)
                            
                            # 润色单条技能
                            prompt = skill._build_single_skill_prompt(line, None)
                            result = skill.agent.generate(prompt, deepThinking=False)
                            polished = skill._clean_single_result(result)
                            
                            # 检查清理后的内容是否有效（不能是列表标记、不能是空内容）
                            is_valid = (
                                polished and 
                                len(polished) > 10 and 
                                not polished.strip().startswith('-') and
                                not polished.strip().startswith('[') and
                                not polished.strip() in ['改写失败', ''] and
                                'assistant' not in polished.lower()
                            )
                            
                            if is_valid:
                                polished_skills_lines.append(polished)
                                # 发送该条技能的优化结果到思考过程
                                yield json.dumps({"type": "thought", "content": f"  ✓ {polished}\n"}, ensure_ascii=False) + "\n"
                            else:
                                # 清理结果无效，使用原始内容
                                polished_skills_lines.append(line)
                                yield json.dumps({"type": "thought", "content": f"  ✓ {line}\n"}, ensure_ascii=False) + "\n"
                            await asyncio.sleep(0.05)
                    
                    # 处理项目
                    polished_projects_list = []
                    if projects_content and projects_content != '无':
                        yield json.dumps({"type": "thought", "content": "\n【正在优化项目经历】\n"}, ensure_ascii=False) + "\n"
                        await asyncio.sleep(0.1)
                        
                        project_paragraphs = [p.strip() for p in projects_content.split('\n\n') if p.strip()]
                        total_projects = len(project_paragraphs)
                        
                        for idx, paragraph in enumerate(project_paragraphs):
                            if not paragraph:
                                continue
                            
                            # 发送进度到思考过程
                            yield json.dumps({"type": "thought", "content": f"  处理第 {idx+1}/{total_projects} 个项目...\n"}, ensure_ascii=False) + "\n"
                            await asyncio.sleep(0.05)
                            
                            # 润色单个项目
                            prompt = skill._build_single_project_prompt(paragraph, None)
                            result = skill.agent.generate(prompt, deepThinking=False)
                            polished = skill._clean_single_result(result)
                            
                            # 检查清理后的内容是否有效
                            is_valid = (
                                polished and 
                                len(polished) > 20 and 
                                not polished.strip().startswith('-') and
                                not polished.strip().startswith('[') and
                                not polished.strip() in ['改写失败', ''] and
                                'assistant' not in polished.lower()
                            )
                            
                            if is_valid:
                                polished_projects_list.append(polished)
                                # 发送该项目的优化结果到思考过程（只显示前50字）
                                preview = polished[:50] + "..." if len(polished) > 50 else polished
                                yield json.dumps({"type": "thought", "content": f"  ✓ {preview}\n"}, ensure_ascii=False) + "\n"
                            else:
                                # 清理结果无效，使用原始内容
                                polished_projects_list.append(paragraph)
                                preview = paragraph[:50] + "..." if len(paragraph) > 50 else paragraph
                                yield json.dumps({"type": "thought", "content": f"  ✓ {preview}\n"}, ensure_ascii=False) + "\n"
                            await asyncio.sleep(0.05)
                    
                    # 发送完成标记到思考过程
                    yield json.dumps({"type": "thought", "content": "\n" + "="*50 + "\n优化完成！\n"}, ensure_ascii=False) + "\n"
                    
                    # 组合最终结果
                    skills_result = '\n'.join(polished_skills_lines) if polished_skills_lines else '无'
                    projects_result = '\n\n'.join(polished_projects_list) if polished_projects_list else '无'
                    polished_content = f"【技能】\n{skills_result}\n\n【项目经历】\n{projects_result}"
                    
                    # 发送最终结果到优化建议（token 类型）
                    yield json.dumps({"type": "token", "content": polished_content}, ensure_ascii=False) + "\n"
                    yield json.dumps({"type": "complete"}, ensure_ascii=False) + "\n"
                except Exception as e:
                    logger.error(f"[{call_time}] 流式处理出错：{e}")
                    yield json.dumps({"type": "thought", "content": f"\n❌ 处理出错：{str(e)}\n"}, ensure_ascii=False) + "\n"
                    yield json.dumps({"type": "token", "content": f"处理失败：{str(e)}"}, ensure_ascii=False) + "\n"
                    yield json.dumps({"type": "complete"}, ensure_ascii=False) + "\n"
            
            return StreamingResponse(event_stream(), media_type="application/json")
            
        except Exception as e:
            logger.error(f"[{call_time}] 简历润色 skill 执行失败：{e}")
            full_response = f"""【原始内容】
{skills if skills else '无'}

{projects if projects else '无'}

（注：简历润色处理失败，显示原始内容）"""
            return {"response": full_response, "file_path": file_path}
        
    except Exception as err:
        logger.error(f"[{call_time}] Upload file error: {err}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File upload failed")
    finally:
        # 清理临时文件
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


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


# ==================== 多 Agent 简历优化 API ====================

from pydantic import BaseModel
from typing import Optional
from agents.langgraph.resume_agents import get_resume_workflow


class ResumeOptimizeRequest(BaseModel):
    """简历优化请求"""
    resume: str
    jd: Optional[str] = None
    position_type: Optional[str] = None


class ResumeOptimizeResponse(BaseModel):
    """简历优化响应"""
    success: bool
    overall_score: Optional[Dict[str, Any]]
    scores: Optional[Dict[str, Any]]
    suggestions: Optional[List[Dict[str, Any]]]
    optimized_resume: Optional[str]
    match_analysis: Optional[Dict[str, Any]]
    error: Optional[str]


@router.post("/resume/optimize", response_model=ResumeOptimizeResponse)
async def resume_optimize(request: ResumeOptimizeRequest):
    """
    多 Agent 简历优化接口
    
    使用 3 个 Agent 协作完成简历优化：
    1. ResumeScoreAgent - 简历多维度评分
    2. JDMatchAgent - JD 关键词匹配与优化建议
    3. ResumePolishAgent - 简历润色
    
    Args:
        resume: 简历内容
        jd: 目标岗位 JD（可选）
        position_type: 岗位类型（可选）
    
    Returns:
        包含评分、建议、优化后简历的完整结果
    """
    try:
        logger.info(f"[ResumeOptimize] 开始简历优化，简历长度: {len(request.resume)}")
        
        # 获取工作流实例
        workflow = get_resume_workflow()
        
        # 执行优化
        result = workflow.optimize(
            resume=request.resume,
            jd=request.jd,
            position_type=request.position_type
        )
        
        logger.info(f"[ResumeOptimize] 优化完成，评分: {result.get('overall_score', {}).get('score', 0)}")
        
        return ResumeOptimizeResponse(**result)
        
    except Exception as e:
        logger.error(f"[ResumeOptimize] 优化失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"简历优化失败: {str(e)}"
        )


@router.post("/resume/optimize/stream")
async def resume_optimize_stream(request: Request, data: ResumeOptimizeRequest):
    """
    多 Agent 简历优化流式接口
    
    流式返回每个 Agent 的执行结果：
    1. type="score" - 评分结果
    2. type="suggestions" - 优化建议
    3. type="polished" - 润色后的简历
    4. type="complete" - 全部完成
    """
    import time
    call_time = time.strftime('%H:%M:%S')
    logger.info(f"[{call_time}] Resume optimize stream started")
    
    async def event_stream():
        try:
            workflow = get_resume_workflow()
            
            for event in workflow.optimize_stream(
                resume=data.resume,
                jd=data.jd,
                position_type=data.position_type
            ):
                # 检查客户端是否断开
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected")
                    break
                
                # 发送事件
                yield json.dumps(event, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)
            
            logger.info(f"[{call_time}] Resume optimize stream completed")
            
        except Exception as e:
            logger.error(f"[{call_time}] Resume optimize stream error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"
    
    return StreamingResponse(event_stream(), media_type="application/json")
