"""
Local Provider
"""

from __future__ import annotations
import time
import threading
import ctypes
from pathlib import Path
from typing import Optional, List, Dict, Generator, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import torch
from logger import get_logger
from services.tracing import start_span, update_current_generation
from ...core.base_provider import BaseProvider
from .streamer import generate_stream

try:
    from transformers import pipeline, BitsAndBytesConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pipeline = None
    BitsAndBytesConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

logger = get_logger(__name__)


class LocalProvider(BaseProvider):

    def __init__(self, model_name: str = "model_serving"):
        """Initialize local model provider."""
        self._model_name = model_name
        self.local_model_path = None
        self.transformers_pipeline = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._stop_generation = False
        self._streamer = None
        self._generate_thread = None

        # 优先尝试在本地模型目录加载
        model_path = Path(model_name)
        if not model_path.is_absolute():
            root = Path(__file__).resolve().parents[5]
            local_candidate = root / "ai" / "models" / model_name
            if local_candidate.exists():
                model_path = local_candidate

        if model_path.exists():
            self.local_model_path = str(model_path)
            self._load_model()
        else:
            raise FileNotFoundError(
                f"本地模型路径不存在：{model_path}. 仅支持 ai/models 目录下模型。"
            )

    @property
    def model_name(self) -> str:
        """返回当前使用的模型名称"""
        return self._model_name

    @property
    def device(self) -> str:
        """返回当前设备信息"""
        return self._device

    def _load_model(self):
        """加载本地模型"""
        if pipeline is None:
            raise ImportError("transformers 库未安装，无法加载本地模型")

        try:
            logger.info(f"正在加载本地模型：{self.local_model_path}")
            start_time = time.time()

            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path,
                trust_remote_code=True,
                use_fast=True
            )

            # 配置 4bit 量化
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_path,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )

            # 创建 pipeline
            self.transformers_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ 本地模型加载完成，耗时：{elapsed:.2f}秒")

        except Exception as e:
            logger.error(f"本地模型加载失败：{e}")
            raise

    def stop_generation(self):
        """停止当前正在进行的生成任务

        实际停止由 generate_stream 内部循环轮询 _stop_generation 标志实现
        （streamer/生成线程是流式函数的局部变量，无法从外部直接操作）。
        """
        self._stop_generation = True
        logger.info("用户请求停止生成")

    def _is_stop_requested(self) -> bool:
        """检查是否请求停止"""
        return self._stop_generation

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """异步生成文本"""
        import asyncio
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: self.generate(prompt, **kwargs)
            )
        return result

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """从本地模型生成文本"""
        if images:
            raise ValueError("当前仅支持文本输入，本地模型不支持图片参数")

        if self.transformers_pipeline is None:
            raise RuntimeError("本地模型尚未加载，无法执行生成。")

        timeout = kwargs.get('timeout', 600)

        def _generate_task(prompt, **gen_kwargs):
            tokenizer = self.transformers_pipeline.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt")

            max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
            generate_kwargs = {
                **inputs,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": gen_kwargs.get("temperature", 0.35),
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.2,
                "top_p": 0.9,
                "top_k": 40,
                "do_sample": True,
            }

            outputs = self.transformers_pipeline.model.generate(**generate_kwargs)

            # 解码生成的内容（去掉输入prompt部分）
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return result, input_length, len(generated_tokens)

        try:
            with start_span(
                "generate-response",
                as_type="generation",
                input=prompt[:2000],
                metadata={"model": self._model_name, "device": self._device,
                          "provider": "local"},
            ):
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(_generate_task, prompt, **kwargs)
                    result, in_tokens, out_tokens = future.result(timeout=timeout)
                except TimeoutError:
                    # 不等待生成线程结束（shutdown(wait=False)），立即抛出超时
                    raise RuntimeError(f"模型推理超时，超过了 {timeout} 秒的限制")
                finally:
                    executor.shutdown(wait=False)
                update_current_generation(
                    output=result[:2000],
                    usage={"input": in_tokens, "output": out_tokens},
                )
                return result
        except Exception as e:
            raise RuntimeError(f"Local transformers 生成失败：{e}") from e

    def generate_stream(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """流式生成带思考过程的内容"""
        self._stop_generation = False
        self._streamer = None
        self._generate_thread = None

        if self.transformers_pipeline is None:
            yield {"type": "error", "content": "本地模型尚未加载，无法执行生成。"}
            return

        with start_span(
            "generate-response",
            as_type="generation",
            input=prompt[:2000],
            metadata={"model": self._model_name, "device": self._device,
                      "provider": "local", "stream": True},
        ):
            yield from generate_stream(
                pipeline=self.transformers_pipeline,
                prompt=prompt,
                stop_generation_flag=self._is_stop_requested,
                images=images,
                **kwargs
            )

    def generate_with_thoughts(self, prompt: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """流式生成文本，支持思考过程"""
        yield from self.generate_stream(prompt, **kwargs)

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """生成带图片的响应（当前不支持）"""
        raise NotImplementedError("本地模型暂不支持图片输入")
