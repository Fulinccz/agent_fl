from __future__ import annotations

from .base import BaseAgent
import requests
import base64
from typing import Optional, List, Dict, Any

class LocalAgent(BaseAgent):
    """A local (offline) model adapter using Ollama.

    This implementation uses Ollama's HTTP API to interact with local models,
    supporting both text and multimodal inputs (images, videos).
    """

    def __init__(self, model_name: str = "qwen3.5:4b-vl"):
        """Initialize the local agent with Ollama.
        
        Args:
            model_name: The name of the Ollama model to use.
        """
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"Using Ollama model: {self.model_name}")
        print(f"Ollama API URL: {self.ollama_url}")

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text using Ollama model.
        
        Args:
            prompt: The input prompt for the model.
            images: List of base64-encoded images (optional).
            **kwargs: Additional generation parameters.
            
        Returns:
            The generated text.
        """
        # 构建请求数据
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        # 添加图像输入（如果有）
        if images:
            data["images"] = images
        
        # 发送请求到 Ollama API
        try:
            response = requests.post(self.ollama_url, json=data, timeout=60)  # 增加超时时间以处理图像
            response.raise_for_status()  # 检查请求是否成功
            
            # 解析响应
            result = response.json()
            return result.get("response", "")
            
        except requests.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """Generate text using Ollama model with an image.
        
        Args:
            prompt: The input prompt for the model.
            image_path: Path to the image file.
            **kwargs: Additional generation parameters.
            
        Returns:
            The generated text.
        """
        # 读取并编码图像
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # 调用 generate 方法
            return self.generate(prompt, images=[image_data], **kwargs)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return f"Error: {str(e)}"