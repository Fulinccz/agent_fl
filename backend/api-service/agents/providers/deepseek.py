"""DeepSeek Provider（OpenAI 兼容协议）

基于 OpenAI SDK 访问 DeepSeek OpenAPI：
    base_url: https://api.deepseek.com
    模型: deepseek-flash / deepseek-v4-pro（见 https://api-docs.deepseek.com/zh-cn/）

接口与 LocalProvider 对齐：
    generate(prompt, deepThinking=..., max_new_tokens=...) -> str
    generate_stream(prompt, ...) -> Generator[{"type": "token"|"thought"|"error", "content": str}]
    stop_generation()

API Key 优先级：构造参数 > 环境变量 DEEPSEEK_API_KEY（含根目录 .env）
"""
from __future__ import annotations

import os
import threading
from typing import Optional, List, Dict, Generator, Any

from logger import get_logger
from services.tracing import start_span, update_current_generation
from ..core.base_provider import BaseProvider

logger = get_logger(__name__)


class DeepSeekProvider(BaseProvider):
    """DeepSeek 在线模型 Provider

    用法:
        provider = DeepSeekProvider(api_key="sk-xxx", model="deepseek-flash")
        text = provider.generate("你好")
        for chunk in provider.generate_stream("你好"):
            print(chunk)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("需要安装 openai SDK（>=1.0）: pip install openai") from e

        # 配置：显式参数 > config.yaml/env > 代码默认值
        cfg = self._load_cfg()
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or ""
        self._model_name = model or cfg.get("model", "deepseek-flash")
        self._base_url = base_url or cfg.get("base_url", "https://api.deepseek.com")
        self._timeout = timeout if timeout is not None else float(cfg.get("timeout", 120))
        self._max_retries = max_retries if max_retries is not None else int(cfg.get("max_retries", 2))
        self._device = "cloud"

        self._client = OpenAI(
            api_key=self._api_key or "missing",
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

        # 停止控制
        self._stop_event = threading.Event()
        self._current_stream = None
        self._stream_lock = threading.Lock()

        if not self._api_key:
            logger.warning("[DeepSeekProvider] DEEPSEEK_API_KEY 未设置，调用时将返回错误提示")

    @staticmethod
    def _load_cfg() -> Dict[str, Any]:
        """从业务配置读取 deepseek 小节（失败时用默认值）"""
        try:
            from services.app_config import load_app_config
            cfg = load_app_config().models.deepseek
            return {
                "model": cfg.model,
                "base_url": cfg.base_url,
                "timeout": cfg.timeout,
                "max_retries": cfg.max_retries,
            }
        except Exception as e:
            logger.warning(f"[DeepSeekProvider] 读取业务配置失败，使用默认值: {e}")
            return {}

    # ---------------- 属性 ----------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def api_key(self) -> Optional[str]:
        """API Key（脱敏）"""
        if self._api_key and len(self._api_key) > 8:
            return f"{self._api_key[:4]}...{self._api_key[-4:]}"
        return self._api_key or None

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _build_messages(prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _build_kwargs(self, **kwargs) -> Dict[str, Any]:
        """将本地 Provider 的调用参数映射为 DeepSeek API 参数"""
        payload: Dict[str, Any] = {}
        max_new_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_tokens")
        if max_new_tokens:
            payload["max_tokens"] = int(max_new_tokens)
        if kwargs.get("temperature") is not None:
            payload["temperature"] = float(kwargs["temperature"])
        # deepseek-flash 默认开启思考（reasoning 消耗 max_tokens 且正文可能为空），
        # 必须显式声明：deepThinking=True 启用深度思考，False/缺省直出正文
        thinking = "enabled" if kwargs.get("deepThinking") else "disabled"
        payload["extra_body"] = {"thinking": {"type": thinking}}
        return payload

    def _require_key(self) -> Optional[Dict[str, Any]]:
        """API Key 校验，未设置时返回错误事件"""
        if not self._api_key:
            return {
                "type": "error",
                "content": "DeepSeek API Key 未设置，请在仓库根目录 .env 中配置 DEEPSEEK_API_KEY",
            }
        return None

    # ---------------- 同步生成 ----------------

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        if images:
            raise ValueError("当前仅支持文本输入，DeepSeek Provider 不支持图片参数")

        err = self._require_key()
        if err:
            raise RuntimeError(err["content"])

        self._stop_event.clear()
        try:
            with start_span(
                "generate-response",
                as_type="generation",
                input=prompt[:2000],
                metadata={"model": self._model_name, "base_url": self._base_url,
                          "provider": "deepseek"},
            ):
                resp = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=self._build_messages(prompt),
                    stream=False,
                    timeout=60.0,
                    **self._build_kwargs(**kwargs),
                )
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    update_current_generation(usage={
                        "input": getattr(usage, "prompt_tokens", 0) or 0,
                        "output": getattr(usage, "completion_tokens", 0) or 0,
                    })
                update_current_generation(output=content[:2000])
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 调用失败：{e}") from e

    # ---------------- 流式生成 ----------------

    def generate_stream(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式生成，chunk 契约与本地 Provider 一致：token / thought / error"""
        if images:
            yield {"type": "error", "content": "本地模型暂不支持图片输入"}
            return

        err = self._require_key()
        if err:
            yield err
            return

        self._stop_event.clear()
        try:
            with start_span(
                "generate-response",
                as_type="generation",
                input=prompt[:2000],
                metadata={"model": self._model_name, "provider": "deepseek",
                          "stream": True},
            ):
                stream = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=self._build_messages(prompt),
                    stream=True,
                    stream_options={"include_usage": False},
                    **self._build_kwargs(**kwargs),
                )
                with self._stream_lock:
                    self._current_stream = stream

                collected: List[str] = []
                for chunk in stream:
                    if self._stop_event.is_set():
                        logger.info("[DeepSeekProvider] 用户请求停止生成")
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        yield {"type": "thought", "content": reasoning}
                    if delta.content:
                        collected.append(delta.content)
                        yield {"type": "token", "content": delta.content}

                update_current_generation(output="".join(collected)[:2000])

        except Exception as e:
            if not self._stop_event.is_set():
                yield {"type": "error", "content": f"DeepSeek API 流式调用失败：{e}"}
        finally:
            with self._stream_lock:
                self._current_stream = None

    def generate_with_thoughts(
        self, prompt: str, **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """流式生成（含思考过程），与 generate_stream 等价"""
        yield from self.generate_stream(prompt, **kwargs)

    # ---------------- 停止 ----------------

    def stop_generation(self):
        """停止当前生成：置位标志并关闭流式连接"""
        self._stop_event.set()
        with self._stream_lock:
            stream = self._current_stream
            self._current_stream = None
        if stream is not None:
            try:
                stream.close()
                logger.info("[DeepSeekProvider] 流式连接已关闭")
            except Exception as e:
                logger.warning(f"[DeepSeekProvider] 关闭流式连接失败: {e}")
        logger.info("[DeepSeekProvider] 用户请求停止生成")

    # ---------------- 图片 ----------------

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        raise NotImplementedError("DeepSeek Provider 暂不支持图片输入")
