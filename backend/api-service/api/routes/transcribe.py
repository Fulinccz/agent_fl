"""
语音转写路由
/transcribe

只做「音频 → 文本」一步，不进入 MAS 流水线：
前端录音（MediaRecorder）上传后立即返回转写文本，由用户校对、编辑后再触发生成。
与 /agent/upload_stream 共用同一套上传校验（扩展名/大小/魔数/合规）。
"""

from __future__ import annotations
import asyncio
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse

from logger import get_logger
from services.compliance import check_text
from services.rate_limit import check_rate_limit
from rag.multimodal import AUDIO_EXTS, parse_multimodal, MultimodalDependencyError
from .common import cleanup_file, receive_upload

router = APIRouter(tags=["语音"])
logger = get_logger(__name__)


@router.post(
    "/transcribe",
    summary="语音转写（录音 → 文本）",
    description="""
前端录音（浏览器 MediaRecorder）或本地音频文件上传后，返回转写文本，**不进入** MAS 流水线。

转写由本地 faster-whisper 完成（离线，音频不出本机），临时文件在转写后立即删除。

**请求格式：** `multipart/form-data`
- `file`：音频文件（必填，支持 mp3/wav/m4a/flac/ogg/webm，最大 10MB）

**响应：** `{"text": "...", "modality": "audio"}`
""",
    responses={
        200: {"description": "转写成功"},
        400: {"description": "文件类型不支持、文件损坏或语音依赖缺失"},
        403: {"description": "转写文本未通过合规校验"},
        413: {"description": "文件过大"},
        422: {"description": "未识别到有效语音内容"},
        429: {"description": "请求过于频繁"},
    },
)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    # 限流：与上传同档（每 IP 每 60s 10 次）
    allowed, retry_after, _ = check_rate_limit(request, "/api/transcribe")
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": f"请求过于频繁，请 {retry_after} 秒后重试"},
        )

    received = await receive_upload(request, file, AUDIO_EXTS)
    if received.error is not None:
        return received.error
    file_path = received.path

    try:
        # STT 为 CPU 密集阻塞任务，放入线程池避免卡事件循环
        parsed = await asyncio.to_thread(parse_multimodal, file_path)
    except MultimodalDependencyError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error("语音转写失败：%s", e)
        return JSONResponse(status_code=400, content={"error": f"语音转写失败：{e}"})
    finally:
        # 转写结果已进内存，临时音频立即删除，避免磁盘堆积
        cleanup_file(file_path)

    text = (parsed.get("text") or "").strip()
    if not text:
        return JSONResponse(
            status_code=422,
            content={"error": "没有识别到语音内容，请靠近麦克风、放慢语速后重试"},
        )

    # 转写文本是用户可控输入，与简历正文同一套合规校验
    text_check = check_text(text, "resume")
    if not text_check.allowed:
        return JSONResponse(status_code=text_check.code, content={"error": text_check.reason})

    logger.info("语音转写完成，文本长度：%s", len(text))
    return {"text": text, "modality": "audio"}
