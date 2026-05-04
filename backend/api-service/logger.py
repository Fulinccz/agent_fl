import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional, Dict, Any

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Thread-local storage for trace context
_local = threading.local()


def get_trace_id() -> str:
    """获取当前线程的 trace_id"""
    if not hasattr(_local, "trace_id"):
        _local.trace_id = str(uuid.uuid4())[:16]
    return _local.trace_id


def set_trace_id(trace_id: str):
    """设置当前线程的 trace_id"""
    _local.trace_id = trace_id


def clear_trace_id():
    """清除当前线程的 trace_id"""
    if hasattr(_local, "trace_id"):
        delattr(_local, "trace_id")


class JSONFormatter(logging.Formatter):
    """JSON 格式日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", get_trace_id()),
            "span_id": getattr(record, "span_id", ""),
            "service": "api-service",
            "source": {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            },
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_obj["extra"] = record.extra

        # Add exception info
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """文本格式日志格式化器（开发环境使用）"""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", get_trace_id())
        return (
            f"{self.formatTime(record)} [{record.levelname}] "
            f"[{trace_id}] [{record.name}] {record.getMessage()}"
        )


def setup_logging(
    level: Optional[str] = None,
    json_format: bool = None,
    stream=None,
) -> logging.Logger:
    """初始化全局日志配置"""
    if level is None:
        level = DEFAULT_LOG_LEVEL
    if json_format is None:
        json_format = os.getenv("LOG_FORMAT", "text").lower() == "json"
    if stream is None:
        stream = sys.stdout

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(handler)

    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取命名日志对象"""
    return logging.getLogger(name or __name__)


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
):
    """带上下文的日志记录"""
    trace_id = trace_id or get_trace_id()
    record_extra = {"trace_id": trace_id}
    if extra:
        record_extra["extra"] = extra

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message, extra=record_extra)


def init_app_logger():
    """在应用启动时调用"""
    setup_logging()
    logger = get_logger("api-service")
    logger.info("Logger initialized: level=%s", DEFAULT_LOG_LEVEL)


# 自动初始化
init_app_logger()
