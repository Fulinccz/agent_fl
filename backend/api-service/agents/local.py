from __future__ import annotations

from .base import BaseAgent
from pathlib import Path
from typing import Optional, List

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class LocalAgent(BaseAgent):
    """A local-only model adapter using transformers from ai/models."""

    def __init__(self, model_name: str = "Qwen3___5-4B"):
        """Initialize local model adapter.

        model_name can be:
         - local folder name in ai/models (relative) or absolute path
        """
        self.model_name = model_name
        self.local_model_path = None
        self.transformers_pipeline = None

        # 优先尝试在本地模型目录加载
        model_path = Path(model_name)
        if not model_path.is_absolute():
            # backend/api-service/..  -> repo root
            root = Path(__file__).resolve().parents[3]
            local_candidate = root / "ai" / "models" / model_name
            if local_candidate.exists():
                model_path = local_candidate

        if model_path.exists():
            self.local_model_path = str(model_path)
            if pipeline is None:
                raise RuntimeError("transformers 库未安装，请安装 transformers 并重试")

            try:
                self.transformers_pipeline = pipeline(
                    "text-generation",
                    model=self.local_model_path,
                    tokenizer=self.local_model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
                print(f"Using local transformers model: {self.local_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize transformers local model ({self.local_model_path}): {e}")
        else:
            raise FileNotFoundError(
                f"本地模型路径不存在: {model_path}. 仅支持 ai/models 目录下模型。"
            )

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text from local transformers model only."""
        if images:
            raise ValueError("当前仅支持文本输入，本地模型不支持图片参数")

        if self.transformers_pipeline is None:
            raise RuntimeError("本地模型尚未加载，无法执行生成。")

        try:
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 256),
                "do_sample": kwargs.get("do_sample", False),
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.95),
                **kwargs,
            }
            outputs = self.transformers_pipeline(prompt, **gen_kwargs)
            if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
                return outputs[0]["generated_text"]
            return str(outputs)
        except Exception as e:
            raise RuntimeError(f"Local transformers 生成失败: {e}")

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """当前本地模型仅支持文本推理。"""
        raise NotImplementedError("本地模型当前仅支持文本输入，暂不支持图像生成。")