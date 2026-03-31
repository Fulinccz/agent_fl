from __future__ import annotations
from logger import get_logger
import torch
from .base import BaseAgent
from pathlib import Path
from typing import Optional, List

logger = get_logger(__name__)

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class LocalAgent(BaseAgent):
    """A local-only model adapter using transformers from ai/models."""

    def __init__(self, model_name: str = "model_serving"):
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
                    trust_remote_code=True,
                    device_map="cpu",               # CPU稳定运行
                    dtype=torch.bfloat16,           # 无损精度，速度大幅提升（新版参数名）
                )
                self.transformers_pipeline.model = torch.compile(
                    self.transformers_pipeline.model,
                    mode="max-autotune"
                )
                logger.info("Using local transformers model: %s", self.local_model_path)
            except Exception as e:
                logger.exception("Failed to initialize transformers local model: %s", self.local_model_path)
                raise RuntimeError(f"Failed to initialize transformers local model ({self.local_model_path}): {e}") from e
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
            outputs = self.transformers_pipeline(prompt)
            if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
                return outputs[0]["generated_text"]
            return str(outputs)
        except Exception as e:
            raise RuntimeError(f"Local transformers 生成失败: {e}")

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """当前本地模型仅支持文本推理。"""
        raise NotImplementedError("本地模型当前仅支持文本输入，暂不支持图像生成。")