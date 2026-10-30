"""
简历优化相关路由
/resume/*

主从多智能体协作简历优化流水线：
Master 规划（评分/匹配 并行步骤 → 润色）→ 执行器执行 → ReAct 反思 → 流式输出。

事件契约与旧版保持一致：score → suggestions → polished → complete，
另附带 intent / plan / step_start / step_result / reflection 过程事件。
"""

from __future__ import annotations
import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from starlette.concurrency import iterate_in_threadpool

from logger import get_logger
from services.compliance import check_text, check_file_magic
from services.rate_limit import check_rate_limit
from .common import ResumeOptimizeRequest, build_preset_intent

router = APIRouter(tags=["简历"])
logger = get_logger(__name__)

# 导出文件登记表：token → 文件信息。文件留在导出目录，按 TTL 惰性清理，
# 保证前端拿到的地址在一段时间内始终可打开/预览。
_EXPORTS: dict[str, dict] = {}
_EXPORT_TTL_SECONDS = 2 * 3600


@router.post(
    "/optimize/stream",
    summary="流式简历优化",
    description="""
启动主从多智能体协作的简历优化流程，流式返回各阶段结果。

**执行过程：**
1. Master 规划：评分、JD 匹配（可并行）→ 简历润色
2. 各步骤由执行器执行，ReAct 反思器审查结果（失败自动重试/重规划）

**事件契约（与旧版一致）：**
- `score` - 简历评分（匹配度、完整性、专业度）
- `suggestions` - 优化建议（具体改进点）
- `polished` - 润色后的完整简历（partial=true 为流式中间结果）
- `complete` - 全部完成

**过程事件（新增，前端可忽略）：**
- `intent` / `plan` / `step_start` / `step_result` / `reflection`

**请求参数：**
- `resume`（必填）：简历原文
- `jd`（可选）：目标职位描述，用于针对性优化
- `position_type`（可选）：职位类型

**响应格式：** `application/json`（每行一个 JSON 对象）
    """,
    responses={
        200: {"description": "流式响应"},
        500: {"description": "优化流程错误"},
    },
)
async def resume_optimize_stream(request: Request, data: ResumeOptimizeRequest):
    call_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"[{call_time}] Resume optimize stream started")

    # 合规风控：限流 + 内容校验（在建立流式连接前拦截）
    allowed, retry_after, _ = check_rate_limit(request, "/api/resume/optimize/stream")
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": f"请求过于频繁，请 {retry_after} 秒后重试"},
        )
    for kind, text in (("resume", data.resume or ""), ("jd", data.jd or "")):
        result = check_text(text, kind)
        if not result.allowed:
            logger.warning(f"[{call_time}] 合规拦截 kind={kind}: {result.reason}")
            return JSONResponse(status_code=result.code, content={"error": result.reason})

    # 预置意图：resume 路由的场景意图明确，跳过意图识别；
    # 页面入口（分析/生成/面试）优先，旧调用按 intent 快捷操作映射，未知值回落全流程优化
    preset_intent = build_preset_intent(data.intent, data.page)

    async def event_stream():
        try:
            from api.routes.deps import get_master_orchestrator
            from services.page_profiles import apply_page_context
            orchestrator = get_master_orchestrator()

            context = {
                "resume": data.resume,
                "jd": data.jd,
                "position_type": data.position_type,
            }
            # 页面档案：计划步骤 + 执行器白名单 + 提示词指令
            apply_page_context(context, data.page)
            # 模型切换：透传到 MAS 黑板，执行器按此选择 Provider
            if data.provider:
                context["provider"] = data.provider
            if data.model:
                context["model"] = data.model

            async for event in iterate_in_threadpool(
                orchestrator.run_stream("", context, preset_intent)
            ):
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected")
                    break

                yield json.dumps(event, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            logger.info(f"[{call_time}] Resume optimize stream completed (MAS)")

        except Exception as e:
            logger.error(f"[{call_time}] MAS optimize error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")


@router.post(
    "/export",
    summary="导出优化简历文档并登记访问地址",
    description="""
将优化后的简历文本生成 .docx 文档，落盘后登记访问地址返回。

**双路径策略：**
- 携带 `file`（用户上传过的原始 docx）→ 复制原文件按内容对齐改写，
  保留原文档的图标、图片、表格、页面设置与样式
- 不携带 `file` → 按内置简历模板重新排版生成

**响应：**
- `url`：docx 文件地址（可直接打开/下载）
- `preview_url`：只读 HTML 预览地址（服务端渲染，不可编辑）
- `filename` / `size`：文档名与字节数

导出文件按 2 小时 TTL 惰性清理；源文件（上传副本）生成后立即删除。

**请求格式：** `multipart/form-data`
- `text`：优化后的简历文本（必填）
- `file`：用户上传的原始文件（可选，仅 .docx 支持改写路径）
    """,
    responses={
        200: {"description": "导出成功，返回文件地址"},
        400: {"description": "内容为空或合规拦截"},
        429: {"description": "请求过于频繁"},
        500: {"description": "文档生成失败"},
    },
)
async def export_resume(
    request: Request,
    text: str = Form(...),
    file: UploadFile | None = File(None),
):
    call_time = datetime.now().strftime('%H:%M:%S')

    allowed, retry_after, _ = check_rate_limit(request, "/api/resume/export")
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": f"导出过于频繁，请 {retry_after} 秒后重试"},
        )
    result = check_text(text or "", "resume")
    if not result.allowed:
        return JSONResponse(status_code=result.code, content={"error": result.reason})

    source_path = None
    try:
        from services.resume_document import generate_resume_docx

        # 可选原文件：保存到临时目录供改写路径使用
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in {".docx"}:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"改写路径仅支持 .docx 原文件，收到 {ext}"},
                )
            content = await file.read()
            if len(content) > 10 * 1024 * 1024:
                return JSONResponse(status_code=400, content={"error": "原文件过大（>10MB）"})
            magic = check_file_magic(content, file.filename)
            if not magic.allowed:
                return JSONResponse(status_code=magic.code, content={"error": magic.reason})
            from services.resume_document import sanitize_filename
            safe_name = sanitize_filename(file.filename, default="resume.docx")
            source_path = os.path.join(
                tempfile.gettempdir(),
                f"fulin_src_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{safe_name}",
            )
            with open(source_path, "wb") as f:
                f.write(content)

        # python-docx 生成/CPU 密集操作放入线程池
        export = await asyncio.to_thread(
            generate_resume_docx, text, source_path, file.filename if file else None
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"[{call_time}] 文档生成失败: {e}")
        return JSONResponse(status_code=500, content={"error": f"文档生成失败：{e}"})
    finally:
        # 上传副本只服务于本次改写，不留在服务端（成功/失败都清理）
        _remove_quiet(source_path)

    token = _register_export(export)
    logger.info(f"[{call_time}] 文档导出成功 mode={export.mode} file={export.filename}")

    return {
        "url": f"/api/resume/exports/{token}",
        "preview_url": f"/api/resume/exports/{token}/preview",
        "filename": export.filename,
        "size": export.size,
    }


@router.get(
    "/exports/{token}",
    summary="下载已导出的简历文档",
    description="按导出时返回的地址下载 docx；token 过期（2 小时）后返回 404。",
)
async def download_export(token: str):
    entry = _prune_exports().get(token)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": "文件不存在或已过期"})
    return FileResponse(
        entry["path"],
        filename=entry["filename"],
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get(
    "/exports/{token}/preview",
    summary="只读预览已导出的简历文档",
    description="服务端把 docx 渲染为只读 HTML，供前端沙箱 iframe 展示，不提供编辑入口。",
)
async def preview_export(token: str):
    entry = _prune_exports().get(token)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": "文件不存在或已过期"})

    from services.resume_document import render_docx_preview_html

    try:
        html = await asyncio.to_thread(render_docx_preview_html, entry["path"])
    except Exception as e:
        logger.error(f"预览渲染失败 token={token}: {e}")
        return JSONResponse(status_code=500, content={"error": f"预览渲染失败：{e}"})
    # sandbox iframe + 严格 CSP：预览页只渲染，不执行脚本
    return HTMLResponse(
        html,
        headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"},
    )


def _register_export(export) -> str:
    """登记导出文件并返回可访问 token（惰性清理过期项）"""
    _prune_exports()
    token = uuid.uuid4().hex
    _EXPORTS[token] = {
        "path": export.path,
        "filename": export.filename,
        "created": time.time(),
    }
    return token


def _prune_exports() -> dict[str, dict]:
    """清理超过 TTL 的导出文件与登记项"""
    now = time.time()
    for token, entry in list(_EXPORTS.items()):
        if now - entry["created"] < _EXPORT_TTL_SECONDS:
            continue
        _remove_quiet(entry["path"])
        _EXPORTS.pop(token, None)
    return _EXPORTS


def _remove_quiet(path: str | None):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
