"""
流式生成器
处理模型流式输出
"""

from __future__ import annotations
import time
import threading
from typing import Dict, Any, Generator, Optional, List
from logger import get_logger
from .token_processor import TokenProcessor
from .utils import check_stop_patterns

logger = get_logger(__name__)


def generate_stream(
    pipeline,
    prompt: str,
    stop_generation_flag: callable,
    images: Optional[List[str]] = None,
    **kwargs
) -> Generator[Dict[str, Any], None, None]:
    """
    流式生成带思考过程的内容
    
    Args:
        pipeline: transformers pipeline
        prompt: 输入提示
        stop_generation_flag: 停止生成标志函数
        images: 图片列表（当前不支持）
        **kwargs: 生成参数
    
    Yields:
        包含 type 和 content 的字典
    """
    if images:
        yield {"type": "error", "content": "本地模型暂不支持图片输入"}
        return
    
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        yield {"type": "error", "content": "transformers 库版本过低，不支持流式输出"}
        return
    
    tokenizer = pipeline.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt")
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=kwargs.get('timeout', 300),
        chunk_size=1
    )
    
    # 构建生成参数
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": kwargs.get("temperature", 0.35),
        "max_new_tokens": kwargs.get("max_new_tokens", 512),
        "repetition_penalty": 1.1,
        "top_p": 0.75,
        "top_k": 20,
        "do_sample": True,
        "use_cache": True,
        "early_stopping": False,
    }
    
    logger.info(f"开始流式生成 - prompt 长度：{len(prompt)} 字符")
    
    # 确保模型处于 eval 模式
    pipeline.model.eval()
    
    # 启动生成线程
    thread = threading.Thread(
        target=pipeline.model.generate,
        kwargs=generate_kwargs,
        daemon=True
    )
    thread.start()
    
    # 初始化处理器
    processor = TokenProcessor(parse_think=kwargs.get('parse_think', True))
    stop_patterns = [
        "\n【简历评分】",
        "\n### 【简历评分】",
        "\n## 【简历评分】",
    ]
    
    token_count = 0
    start_time = time.time()
    last_token_time = time.time()
    
    try:
        for token in streamer:
            current_time = time.time()
            token_count += 1
            
            # 检查停止标志
            if stop_generation_flag():
                logger.info("用户请求停止生成")
                yield {"type": "thought", "content": "用户已中止生成"}
                if hasattr(streamer, 'stop'):
                    streamer.stop()
                break
            
            # 记录首 token 时间
            if token_count == 1:
                logger.info(f"收到第 1 个 token (首 token 耗时：{current_time - start_time:.2f}s)")
                logger.info(f"首 token 内容: {token!r}")
            
            # 检测生成停滞
            token_interval = current_time - last_token_time
            if token_interval > 60 and token_count > 1:
                logger.warning(f"检测到生成停滞（{token_interval:.2f}s 无新 token）")
            
            last_token_time = current_time
            
            # 处理 token
            outputs = processor.process_token(token)
            
            # 记录前10个输出 token
            if token_count <= 10:
                logger.info(f"Token {token_count}: 输入={token!r}, 输出数量={len(outputs)}")
                for i, output in enumerate(outputs):
                    logger.info(f"  输出 {i}: type={output.get('type')}, content={output.get('content')!r}")
            
            for output in outputs:
                yield output
            
            # 检查停止模式
            if check_stop_patterns(processor.full_text, stop_patterns):
                logger.info("检测到停止模式，终止生成")
                if hasattr(streamer, 'stop'):
                    streamer.stop()
                break
    
    except Exception as e:
        logger.error(f"流式生成错误：{e}")
        yield {"type": "error", "content": f"生成错误：{str(e)}"}
    
    finally:
        logger.info(f"流式生成完成，共 {token_count} 个 token")
