

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

重要规则：
1. 禁止添加用户简历中没有的技术栈
2. 不要引用知识库中的具体数字
3. 只输出以下 3 个部分，输出完就停止：

【简历评分】
只给分数（1-100）

【优化建议】
简短列出 2-3 条

【优化结果（参考）】
只优化技能栏和项目描述

"""
    else:
        system_prompt = """请用中文直接输出结果，不要任何思考过程、分析过程或前缀。

重要规则：
1. 禁止添加用户简历中没有的技术栈
2. 不要引用知识库中的具体数字
3. 只输出以下 3 个部分，输出完就停止：

【简历评分】
只给分数（1-100）

【优化建议】
简短列出 2-3 条

【优化结果（参考）】
只优化技能栏和项目描述

"""
    
    # 构建完整 Prompt
    full_prompt = system_prompt + "\n\n"
    
    # 如果有 RAG 上下文，作为参考放在最后（不作为示例）
    if rag_context:
        full_prompt += f"""
---
【参考资料】（仅供学习表达方式，不要模仿格式）
{rag_context}
---
"""
    
    full_prompt += f"\n用户简历：{data.query}"
    
    async def event_stream():
        try:
            item_count = 0
            # 传递 deepThinking 参数，让后端知道是否应该解析 think 标签
            for item in agent.generate_with_thoughts(full_prompt, deepThinking=getattr(data, 'deepThinking', False)):
                item_count += 1
                
                # 记录每个 item 的类型，便于调试
                logger.debug(f"[{call_time}] Yield item #{item_count}: type={item.get('type', 'unknown')}")
                
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
            logger.error(f"[{call_time}] Stream error: %s", e, exc_info=True)
            agent.stop_generation()
        finally:
            # 只在生成未完成时调用 stop_generation（避免重复调用）
            pass
    
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
