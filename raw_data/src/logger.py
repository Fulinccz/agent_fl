"""
日志配置
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "mcp_collector",
    log_level: str = "INFO",
    log_file: str = "logs/collector.log"
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志名称
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_file,
            encoding='utf-8',
            mode='a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 默认日志记录器
logger = setup_logger()
