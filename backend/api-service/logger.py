import logging
import os
import sys
from typing import Optional

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s %(levelname)-8s [%(name)s] %(filename)s:%(lineno)d %(message)s",
)
DEFAULT_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")


def setup_logging(
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    stream=None,
) -> logging.Logger:
    """初始化全局日志配置，返回 root logger。"""
    if level is None:
        level = DEFAULT_LOG_LEVEL
    if fmt is None:
        fmt = DEFAULT_LOG_FORMAT
    if datefmt is None:
        datefmt = DEFAULT_DATE_FORMAT
    if stream is None:
        stream = sys.stdout

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(stream=stream, level=numeric_level, format=fmt, datefmt=datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取命名日志对象。"""
    return logging.getLogger(name or __name__)


def init_app_logger():
    """在应用启动时调用的便利函数。"""
    setup_logging()
    get_logger("api-service").info("Logger initialized: level=%s", DEFAULT_LOG_LEVEL)


# 自动初始化日志配置，避免模块被导入时无日志
init_app_logger()
