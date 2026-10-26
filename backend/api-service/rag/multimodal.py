"""
多模态统一解析入口

将文档 / 图片 / 音频统一解析为结构化文本，接入 Agent 流程：
- 文档（pdf/doc/docx/txt）：复用 rag.document_processor（txt 直接读取）
- 图片（png/jpg/jpeg/bmp/webp）：RapidOCR 本地离线 OCR
- 音频（mp3/wav/m4a/flac/ogg/webm）：faster-whisper 本地离线语音转写
  （webm 为浏览器 MediaRecorder 默认输出格式，供前端录音直传）

设计原则（与 tracing 一致的低耦合）：
- OCR / STT 为可选依赖，懒加载，未安装时抛出带安装指引的友好错误
- 模型默认放在仓库 ai/models/ 下，可用环境变量覆盖路径
- 引擎单例双检锁，FastAPI 并发首调只初始化一次
- OCR / STT 为 CPU 密集任务，调用方应放入线程池（见 upload 路由 asyncio.to_thread）
- 引擎推理接口本身线程安全：onnxruntime InferenceSession.Run 与 ctranslate2
  WhisperModel.transcribe 均支持多线程并发调用，故单例可被多个 to_thread 任务共享
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional, Any

from logger import get_logger
from services.tracing import start_span, update_observation
from .document_processor import parse_resume

logger = get_logger(__name__)

# 仓库根目录：backend/api-service/rag/multimodal.py → parents[3]
_ROOT_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_MODELS_DIR = _ROOT_DIR / "ai" / "models"

DOCUMENT_EXTS = {".pdf", ".doc", ".docx", ".txt"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

ALL_MULTIMODAL_EXTS = DOCUMENT_EXTS | IMAGE_EXTS | AUDIO_EXTS

# OCR 模型目录（rapidocr onnx：det/rec/cls）。目录为空时回退 rapidocr 内置模型
_OCR_MODEL_DIR = Path(os.getenv("MULTIMODAL_OCR_MODEL_DIR", str(_DEFAULT_MODELS_DIR / "rapidocr")))
# OCR 行文本置信度阈值，低于该值视为误识别并丢弃
_OCR_MIN_SCORE = float(os.getenv("MULTIMODAL_OCR_MIN_SCORE", "0.5"))
# STT 模型目录（faster-whisper 本地路径或 size 名称）
_STT_MODEL_PATH = os.getenv(
    "MULTIMODAL_STT_MODEL_PATH",
    str(_DEFAULT_MODELS_DIR / "whisper" / "faster-whisper-small"),
)
# 中文转写引导语：whisper 未给提示时中文常输出繁体（如“經驗/響應”），
# 用简体提示词可稳定输出简体；顺带列出简历领域术语，提升专有名词识别率。
# 实测同一条音频：无提示 → “或是一名有三年經驗的Java後端工程師…”；
# 加提示 → “或是一名有三年经验的Java后端工程师…”。可用环境变量整体覆盖。
_STT_INITIAL_PROMPT = os.getenv(
    "MULTIMODAL_STT_INITIAL_PROMPT",
    "以下是普通话的句子。常用术语：Java、Python、前端工程师、后端工程师、"
    "订单系统、接口响应时间、性能优化、微服务。",
)

# 懒加载单例 + 双检锁
_ocr_engine: Optional[Any] = None
_stt_model: Optional[Any] = None
_ocr_lock = threading.Lock()
_stt_lock = threading.Lock()


class MultimodalDependencyError(RuntimeError):
    """多模态可选依赖未安装时抛出，携带安装指引"""


def detect_modality(file_path: str) -> str:
    """按扩展名判断模态：document / image / audio"""
    ext = Path(file_path).suffix.lower()
    if ext in DOCUMENT_EXTS:
        return "document"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    raise ValueError(f"不支持的文件类型：{ext}")


def parse_multimodal(file_path: str) -> Dict[str, Any]:
    """统一解析入口

    返回：
        {
            "text": str,           # 统一纯文本（图片为 OCR 文本，音频为转写，文档为全文）
            "modality": "document" | "image" | "audio",
            "skills": str | None,  # 仅结构化文档：技能栏
            "projects": str | None,# 仅结构化文档：项目描述
            "metadata": dict,
        }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = path.suffix.lower()
    modality = detect_modality(file_path)
    base_meta = {"ext": ext}

    if modality == "document":
        return _parse_document(path, ext, base_meta)
    if modality == "image":
        text = _ocr_image(file_path)
        return {"text": text, "modality": "image", "skills": None, "projects": None,
                "metadata": {**base_meta, "engine": "rapidocr"}}
    text = _transcribe_audio(file_path)
    return {"text": text, "modality": "audio", "skills": None, "projects": None,
            "metadata": {**base_meta, "engine": "faster-whisper"}}


def _parse_document(path: Path, ext: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """文档解析：txt 直接读取；pdf/doc/docx 走简历结构化解析"""
    if ext == ".txt":
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk", errors="ignore")
        return {"text": text, "modality": "document", "skills": None,
                "projects": None, "metadata": {**meta, "engine": "plaintext"}}

    data = parse_resume(str(path))
    return {
        "text": data.get("full_content") or "",
        "modality": "document",
        "skills": data.get("skills"),
        "projects": data.get("projects"),
        "metadata": {**meta, "engine": "resume_parser"},
    }


# ---------------- 图片 OCR ----------------

def _get_ocr_engine():
    """懒加载 RapidOCR 单例（双检锁）"""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise MultimodalDependencyError(
                "图片 OCR 需要可选依赖 rapidocr-onnxruntime，请安装：\n"
                "  pip install rapidocr-onnxruntime"
            ) from e

        # 若 ai/models/rapidocr 下存在 onnx 模型则使用本地模型，否则用包内置模型。
        # det_model_path / rec_model_path / cls_model_path 为 RapidOCR UpdateParameters
        # 官方白名单支持的 kwargs 键名。
        kwargs: Dict[str, Any] = {}
        if _OCR_MODEL_DIR.exists():
            det = _OCR_MODEL_DIR / "det.onnx"
            rec = _OCR_MODEL_DIR / "rec.onnx"
            cls = _OCR_MODEL_DIR / "cls.onnx"
            if det.exists() and rec.exists():
                kwargs["det_model_path"] = str(det)
                kwargs["rec_model_path"] = str(rec)
                if cls.exists():
                    kwargs["cls_model_path"] = str(cls)
                logger.info("RapidOCR 使用本地模型目录：%s", _OCR_MODEL_DIR)

        _ocr_engine = RapidOCR(**kwargs)
        return _ocr_engine


def _ocr_image(file_path: str) -> str:
    """对图片做 OCR，返回拼接后的文本（过滤低置信度误识别行）"""
    with start_span(
        "multimodal-ocr",
        as_type="tool",
        input={"file": Path(file_path).name},
        metadata={"engine": "rapidocr"},
    ) as obs:
        engine = _get_ocr_engine()
        result, _ = engine(file_path)
        if not result:
            logger.warning("OCR 未识别到文字：%s", file_path)
            update_observation(obs, output={"lines": 0, "text": ""})
            return ""
        # result 元素：[box, text, score]
        raw_lines = [item for item in result if len(item) >= 2 and item[1]]
        lines: list[str] = []
        dropped = 0
        for item in raw_lines:
            score = item[2] if len(item) >= 3 else 1.0
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 1.0
            if score < _OCR_MIN_SCORE:
                dropped += 1
                continue
            lines.append(item[1])
        text = "\n".join(lines).strip()
        logger.info(
            "图片 OCR 完成：%s，识别 %d 行（过滤低置信度 %d 行）",
            Path(file_path).name, len(lines), dropped,
        )
        update_observation(obs, output={"lines": len(lines), "dropped_low_score": dropped,
                                        "text": text[:500]})
        return text


# ---------------- 语音转写 ----------------

def _get_stt_model():
    """懒加载 faster-whisper 单例（双检锁）"""
    global _stt_model
    if _stt_model is not None:
        return _stt_model
    with _stt_lock:
        if _stt_model is not None:
            return _stt_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise MultimodalDependencyError(
                "语音转写需要可选依赖 faster-whisper，请安装：\n"
                "  pip install faster-whisper"
            ) from e

        # 优先使用 ai/models 下的本地模型；不存在则退回 size 名称（触发下载到 HF 缓存）
        model_ref = _STT_MODEL_PATH
        if not Path(_STT_MODEL_PATH).exists():
            logger.warning("本地语音模型目录不存在：%s，回退为自动下载 small 模型", _STT_MODEL_PATH)
            model_ref = "small"

        _stt_model = WhisperModel(model_ref, device="cpu", compute_type="int8")
        return _stt_model


def _decode_to_array(file_path: str, sampling_rate: int = 16000) -> "Any":
    """用 PyAV 把音频解码成 whisper 需要的 float32 单声道数组。

    faster-whisper 1.2.1 内部调用 `av.open(..., metadata_errors="ignore")`，
    而该参数在 av>=19 已被移除（实测 av 19.0.1 直接 TypeError）。
    这里自行解码后把 ndarray 传给 transcribe，绕开上游这段不兼容的解码分支，
    同时不锁定 av 版本。返回约 [-1, 1] 的 float32、16kHz 单声道。
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks: list = []

    def _collect(frames) -> None:
        for frame in frames or []:
            chunks.append(frame.to_ndarray().reshape(-1))

    with av.open(file_path) as container:
        for frame in container.decode(audio=0):
            out = resampler.resample(frame)
            if out is None:
                continue
            _collect(out if isinstance(out, list) else [out])
        # 冲刷重采样器内部缓冲，避免丢掉尾部音频
        _collect(resampler.resample(None))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe_audio(file_path: str, language: Optional[str] = "zh") -> str:
    """语音转写文本"""
    with start_span(
        "multimodal-stt",
        as_type="tool",
        input={"file": Path(file_path).name, "language": language},
        metadata={"engine": "faster-whisper"},
    ) as obs:
        model = _get_stt_model()
        audio = _decode_to_array(file_path)
        if audio.size == 0:
            logger.warning("音频解码结果为空：%s", Path(file_path).name)
            return ""
        segments, info = model.transcribe(
            audio,
            language=language,
            vad_filter=True,
            # 仅中文需要引导（英文/其他语言加中文提示词反而干扰识别）
            initial_prompt=_STT_INITIAL_PROMPT if language == "zh" else None,
        )
        text = "".join(seg.text for seg in segments).strip()
        duration = getattr(info, "duration", 0.0)
        logger.info(
            "语音转写完成：%s，语言=%s，时长=%.1fs",
            Path(file_path).name, getattr(info, "language", language), duration,
        )
        # 持有式更新：STT 在 asyncio.to_thread 中执行，update_current_span 会丢上下文
        update_observation(obs, output={"duration_s": duration, "text": text[:500]})
        return text


def preheat() -> None:
    """启动时后台预热 OCR / STT 引擎（可选依赖缺失时静默跳过）

    首次上传图片/语音时引擎冷加载需 10-30 秒；启动阶段在后台线程完成加载，
    用户首次上传即可直接推理。仅在有可选依赖时执行，失败不影响服务启动。
    """
    def _warm() -> None:
        for name, loader in (("ocr", _get_ocr_engine), ("stt", _get_stt_model)):
            try:
                loader()
                logger.info("多模态引擎预热完成：%s", name)
            except MultimodalDependencyError:
                logger.info("多模态引擎 %s 依赖未安装，跳过预热", name)
            except Exception as e:
                logger.warning("多模态引擎 %s 预热失败（不影响启动）：%s", name, e)

    threading.Thread(target=_warm, name="multimodal-preheat", daemon=True).start()
