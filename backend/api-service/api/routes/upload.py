"""
文件上传相关路由
/upload

支持文件上传、简历解析，解析出的正文直接进入主从多智能体流水线
（与 /resume/optimize/stream 同一套 MAS 编排：评分 → 匹配 → 精简 → 润色），
流式事件契约与粘贴链路完全一致。
"""

from __future__ import annotations
import os
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import iterate_in_threadpool

from logger import get_logger
from services.compliance import check_text, check_file_magic
from services.rate_limit import check_rate_limit
from rag.multimodal import parse_multimodal, ALL_MULTIMODAL_EXTS, MultimodalDependencyError
from agents.mas.state import IntentResult, INTENT_RESUME_OPTIMIZE
from .common import UPLOAD_DIR

router = APIRouter(tags=["上传"])
logger = get_logger(__name__)


def _cleanup_file(file_path: str | None) -> None:
    """删除上传的临时文件（失败不影响主流程，仅记录）"""
    if not file_path:
        return
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        logger.warning("清理临时上传文件失败 %s：%s", file_path, e)


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
    model: str | None = Form(None)
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

    # 文件类型与大小校验
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    CHUNK_SIZE = 1024 * 1024  # 1MB 分块，内存占用与文件大小无关
    MAGIC_PREFIX_SIZE = 512  # 魔数只看文件头，保留前缀即可

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALL_MULTIMODAL_EXTS:
        return JSONResponse(
            status_code=400,
            content={"error": f"不支持的文件类型：{ext}，支持文档/图片/音频：{', '.join(sorted(ALL_MULTIMODAL_EXTS))}"},
        )

    # 声明长度超限直接拒绝（multipart 编码会有少量开销，留 1KB 余量）
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_FILE_SIZE + 1024:
                return JSONResponse(status_code=413, content={"error": "上传内容过大，最大允许 10MB"})
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "非法的 Content-Length 头"})

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 文件名消毒：防 ../ 路径穿越与 Windows 非法字符
        from services.resume_document import sanitize_filename
        safe_name = sanitize_filename(file.filename)
        filename = f"{timestamp}_{safe_name}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # 分块落盘 + 累计硬限：即使客户端不发 Content-Length 也不会被无限读入内存/磁盘
        total_size = 0
        magic_prefix = b""
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    buffer.close()
                    _cleanup_file(file_path)
                    file_path = None
                    return JSONResponse(
                        status_code=413,
                        content={"error": "文件过大，最大允许 10MB"},
                    )
                if len(magic_prefix) < MAGIC_PREFIX_SIZE:
                    magic_prefix += chunk[:MAGIC_PREFIX_SIZE - len(magic_prefix)]
                buffer.write(chunk)

        logger.info(f"[{call_time}] File uploaded: {file_path} ({total_size} bytes)")

        # 魔数校验：防改扩展名伪装（只需文件头前缀）
        magic = check_file_magic(magic_prefix, file.filename or "")
        if not magic.allowed:
            _cleanup_file(file_path)
            file_path = None
            return JSONResponse(status_code=magic.code, content={"error": magic.reason})

        # 用户补充要求 / JD 合规校验
        query = (query or "").strip()
        jd = (jd or "").strip()
        for kind, text in (("text", query), ("jd", jd)):
            result_check = check_text(text, kind)
            if not result_check.allowed:
                _cleanup_file(file_path)
                file_path = None
                return JSONResponse(status_code=result_check.code, content={"error": result_check.reason})

        # 统一多模态解析。OCR/STT/文档解析为 CPU 密集阻塞任务，放入线程池避免卡事件循环
        try:
            parsed = await asyncio.to_thread(parse_multimodal, file_path)
        except MultimodalDependencyError as e:
            _cleanup_file(file_path)
            return JSONResponse(status_code=400, content={"error": str(e)})

        recognized_text = (parsed.get("text") or "").strip()
        if not recognized_text:
            _cleanup_file(file_path)
            return JSONResponse(
                status_code=422,
                content={"error": "未从上传内容中解析到有效文本，请更换更清晰的文件"},
            )

        # 解析出的简历内容合规校验
        text_check = check_text(recognized_text, "resume")
        if not text_check.allowed:
            _cleanup_file(file_path)
            return JSONResponse(status_code=text_check.code, content={"error": text_check.reason})
        logger.info(
            f"[{call_time}] 解析完成 modality={parsed.get('modality')}，文本长度：{len(recognized_text)}"
        )

        # 解析结果已全部进入内存，临时文件立即清理避免磁盘堆积
        _cleanup_file(file_path)
        file_path = None

        # 预置意图：上传简历场景意图明确，跳过意图识别
        preset_intent = IntentResult(
            name=INTENT_RESUME_OPTIMIZE,
            confidence=1.0,
            source="preset",
            reason="上传文件简历优化预置意图",
            is_complex=True,
        )

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

        async def event_stream():
            try:
                from api.routes.deps import get_master_orchestrator
                orchestrator = get_master_orchestrator()

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
        _cleanup_file(file_path)
        return JSONResponse(status_code=500, content={"error": f"文件处理失败：{e}"})
