"""
文件上传相关路由
/upload

支持文件上传、简历解析，解析出的正文直接进入主从多智能体流水线
（与 /resume/optimize/stream 同一套 MAS 编排：评分 → 匹配 → 精简 → 润色），
流式事件契约与粘贴链路完全一致。
"""

from __future__ import annotations
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import iterate_in_threadpool

from logger import get_logger
from services.compliance import check_text
from services.rate_limit import check_rate_limit
from rag.multimodal import parse_multimodal, ALL_MULTIMODAL_EXTS, MultimodalDependencyError
from .common import cleanup_file, receive_upload, build_preset_intent

router = APIRouter(tags=["上传"])
logger = get_logger(__name__)


@router.post(
    "/agent/upload_stream",
    summary="上传文件并流式优化（MAS 流水线）",
    description="""
上传文件（简历 PDF/DOCX/图片/语音/TXT），自动解析后进入主从多智能体流水线。

**处理流程：**
1. 保存并魔数校验上传文件
2. 多模态解析提取简历正文（OCR/STT/文档解析）
3. 正文进入 MAS：评分 → JD 匹配 →（超长则精简）→ 流式润色

**请求格式：** `multipart/form-data`
- `file`：上传的文件（必填）
- `query`：用户补充要求（可选，作为润色附加指令）
- `jd`：目标岗位 JD（可选）
- `provider` / `model`：模型切换（可选）

**响应事件：** 与 /resume/optimize/stream 完全一致
（score / suggestions / condensed / polished / complete / error 及过程事件）。
    """,
    responses={
        200: {"description": "流式响应"},
        400: {"description": "参数错误或文件解析失败"},
        429: {"description": "请求过于频繁"},
    },
)
async def upload_file_stream(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(""),
    jd: str | None = Form(None),
    provider: str = Form("local"),
    model: str | None = Form(None),
    intent: str = Form(""),
    page: str = Form("")
):
    call_time = datetime.now().strftime('%H:%M:%S')
    file_path = None

    # 限流
    allowed, retry_after, _ = check_rate_limit(request, "/api/agent/upload_stream")
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": f"请求过于频繁，请 {retry_after} 秒后重试"},
        )

    # 文件类型与大小校验、分块落盘、魔数校验（与 /transcribe 共用同一套实现）
    received = await receive_upload(request, file, ALL_MULTIMODAL_EXTS)
    if received.error is not None:
        return received.error
    file_path = received.path
    logger.info(f"[{call_time}] File uploaded: {file_path} ({received.size} bytes)")

    try:
        # 用户补充要求 / JD 合规校验
        query = (query or "").strip()
        jd = (jd or "").strip()
        for kind, text in (("text", query), ("jd", jd)):
            result_check = check_text(text, kind)
            if not result_check.allowed:
                cleanup_file(file_path)
                file_path = None
                return JSONResponse(status_code=result_check.code, content={"error": result_check.reason})

        # 统一多模态解析。OCR/STT/文档解析为 CPU 密集阻塞任务，放入线程池避免卡事件循环
        try:
            parsed = await asyncio.to_thread(parse_multimodal, file_path)
        except MultimodalDependencyError as e:
            cleanup_file(file_path)
            return JSONResponse(status_code=400, content={"error": str(e)})

        recognized_text = (parsed.get("text") or "").strip()
        if not recognized_text:
            cleanup_file(file_path)
            return JSONResponse(
                status_code=422,
                content={"error": "未从上传内容中解析到有效文本，请更换更清晰的文件"},
            )

        # 解析出的简历内容合规校验
        text_check = check_text(recognized_text, "resume")
        if not text_check.allowed:
            cleanup_file(file_path)
            return JSONResponse(status_code=text_check.code, content={"error": text_check.reason})
        logger.info(
            f"[{call_time}] 解析完成 modality={parsed.get('modality')}，文本长度：{len(recognized_text)}"
        )

        # 解析结果已全部进入内存，临时文件立即清理避免磁盘堆积
        cleanup_file(file_path)
        file_path = None

        # 预置意图：上传简历场景意图明确，跳过意图识别；
        # 页面入口（分析/生成/面试）优先，旧调用按 intent 快捷操作映射，未知值回落全流程优化
        preset_intent = build_preset_intent(intent, page)

        context: dict = {"resume": recognized_text}
        if jd:
            context["jd"] = jd
        if query:
            # 用户补充要求透传到润色步骤
            context["extra_instruction"] = query
        if provider:
            context["provider"] = provider
        if model:
            context["model"] = model
        from services.page_profiles import apply_page_context
        apply_page_context(context, page)

        async def event_stream():
            try:
                from api.routes.deps import get_master_orchestrator
                orchestrator = get_master_orchestrator()

                # 解析结果先回填前端输入框：三个页面共用同一份简历文本，
                # 用户切页后不必重新上传
                yield json.dumps(
                    {"type": "resume_parsed", "data": {"text": recognized_text}},
                    ensure_ascii=False,
                ) + "\n"

                async for event in iterate_in_threadpool(
                    orchestrator.run_stream("", context, preset_intent)
                ):
                    if await request.is_disconnected():
                        logger.info(f"[{call_time}] Upload client disconnected")
                        break
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0)

                logger.info(f"[{call_time}] Upload optimize stream completed (MAS)")

            except Exception as e:
                logger.error(f"[{call_time}] Upload MAS error: {e}")
                yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/json")

    except Exception as e:
        logger.error(f"[{call_time}] Upload error: {e}")
        cleanup_file(file_path)
        return JSONResponse(status_code=500, content={"error": f"文件处理失败：{e}"})
