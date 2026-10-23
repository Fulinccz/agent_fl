"""
GGUF Provider（llama-server 子进程 + OpenAI 兼容 HTTP）

替代 transformers fp32 路径的本地推理引擎：
- 内存：Q4_K_M 量化 4B 模型约 2.6GB（fp32 权重约 16GB），且推理在独立进程
- 速度：llama.cpp CPU 内核远快于 torch 对 Qwen3.5 混合架构的慢速回退实现
- 二进制：llama-cpp-pydist 自带 llama-b*-bin-win-cpu-x64.zip，部署后得到 llama-server.exe

架构：首次使用时解压二进制 → 启动 llama-server 子进程（127.0.0.1:8091）→
本类作为 OpenAI 兼容 HTTP 客户端（/v1/chat/completions，支持流式 SSE）。

输出契约与 BaseProvider 完全一致：
generate(prompt, images=None, **kwargs) -> str
generate_stream(...) -> Generator[{"type": "token"|"thought"|"error", "content": ...}]
stop_generation()
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

import requests

from logger import get_logger
from services.tracing import start_span, update_observation
from ...core.base_provider import BaseProvider
from .token_processor import TokenProcessor
from .utils import check_stop_patterns

logger = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SERVER_HOST = "127.0.0.1"
_SERVER_PORT = 8091
_HEALTH_TIMEOUT_S = 120  # 首次加载模型到 RAM 需要数十秒

# 进程级单例：避免多个 Provider 实例重复拉起 server
_server_lock = threading.Lock()
_server_proc: Optional[subprocess.Popen] = None


def _deploy_llama_server() -> Path:
    """从 llama-cpp-pydist 包内 zip 部署 llama-server 二进制，返回 exe 路径"""
    target_dir = Path(__file__).resolve().parents[5] / "ai" / "models" / "gguf" / "llama_bin"
    server_exe = target_dir / "llama-server.exe"
    if server_exe.exists():
        return server_exe

    import llama_cpp  # llama-cpp-pydist
    binaries_dir = Path(llama_cpp.__file__).parent / "binaries"
    zips = list(binaries_dir.glob("llama-*-win-cpu-x64.zip"))
    if not zips:
        raise FileNotFoundError(f"llama-cpp-pydist 包内未找到 Windows 二进制 zip：{binaries_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"部署 llama.cpp 二进制：{zips[0].name} -> {target_dir}")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(target_dir)

    # zip 内可能带一层目录，找到真正的 llama-server.exe
    candidates = list(target_dir.rglob("llama-server.exe"))
    if not candidates:
        raise FileNotFoundError(f"解压后未找到 llama-server.exe：{target_dir}")
    return candidates[0]


def _wait_health(timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{_SERVER_HOST}:{_SERVER_PORT}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _ensure_server(model_path: str, n_ctx: int = 4096) -> None:
    """确保 llama-server 已运行并加载了目标模型（进程级单例）"""
    global _server_proc
    with _server_lock:
        if _server_proc is not None and _server_proc.poll() is None:
            return  # 已由当前进程拉起

        # 尝试复用已运行的健康 server（例如上次异常退出遗留的孤儿进程）
        if _wait_health(timeout_s=1):
            logger.info(f"复用已运行的 llama-server (port={_SERVER_PORT})")
            return

        server_exe = _deploy_llama_server()
        threads = max(4, (os.cpu_count() or 8) // 2)
        cmd = [
            str(server_exe),
            "-m", model_path,
            "--host", _SERVER_HOST,
            "--port", str(_SERVER_PORT),
            "-c", str(n_ctx),
            "-t", str(threads),
            "--jinja",
            "--no-webui",
        ]
        logger.info(f"启动 llama-server：{' '.join(cmd)}")
        _server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if not _wait_health(timeout_s=_HEALTH_TIMEOUT_S):
            raise RuntimeError(
                f"llama-server 健康检查超时（{_HEALTH_TIMEOUT_S}s），模型加载失败或端口被占用"
            )
        atexit.register(_shutdown_server)
        logger.info(f"✅ llama-server 就绪 (port={_SERVER_PORT})")


def _shutdown_server():
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None


class GGUFProvider(BaseProvider):

    def __init__(self, model_name: str = "gguf/Qwen3.5-4B-Q4_K_M.gguf"):
        self._model_name = model_name
        # 活动流式请求的停止 Event 集合（请求级，避免并发流互相复位标志）
        self._stop_events: "set[threading.Event]" = set()
        self._stop_lock = threading.Lock()

        model_path = Path(model_name)
        if not model_path.is_absolute():
            root = Path(__file__).resolve().parents[5]
            model_path = root / "ai" / "models" / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF 模型文件不存在：{model_path}。仅支持 ai/models 目录下的模型。"
            )
        self.local_model_path = str(model_path)
        _ensure_server(self.local_model_path)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        """返回当前设备信息（llama.cpp CPU 推理）"""
        return "cpu"

    def stop_generation(self):
        """停止所有活动流式生成：置位各请求自己的 Event（由 generate_stream 轮询）"""
        with self._stop_lock:
            events = list(self._stop_events)
        for event in events:
            event.set()
        if events:
            logger.info(f"已请求停止 {len(events)} 个流式生成")

    # ---- 内部：OpenAI 兼容调用 ----

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _payload(self, prompt: str, stream: bool, **kwargs) -> Dict[str, Any]:
        """构造 chat/completions 请求体；默认关闭思考模式（任务型 prompt 不需要）

        Qwen3.5 默认生成 <think> 块再输出正文，意图分类/JSON 抽取/润色等任务链路
        不需要思考，白耗 token。chat_template_kwargs 由 llama-server 的 --jinja 模式支持。
        """
        enable_thinking = bool(kwargs.get("deepThinking", False))
        payload: Dict[str, Any] = {
            "model": "local",
            "messages": self._build_messages(prompt),
            "max_tokens": int(kwargs.get("max_new_tokens", 128) or 128),
            "temperature": float(kwargs.get("temperature", 0.35) or 0.35),
            "top_p": 0.9,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if stream:
            # 最后一个 chunk（choices 为空）携带 usage，供 Langfuse 记录 token 用量
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _strip_think(text: str) -> str:
        return _THINK_RE.sub("", text).strip()

    def _post(self, payload: Dict[str, Any], timeout: float) -> requests.Response:
        resp = requests.post(
            f"http://{_SERVER_HOST}:{_SERVER_PORT}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            stream=payload.get("stream", False),
        )
        # llama-server 响应未指定 charset，requests 默认按 ISO-8859-1 解码，
        # 导致流式 iter_lines(decode_unicode=True) 中文乱码。强制 UTF-8。
        resp.encoding = "utf-8"
        return resp

    # ---- 公开接口（契约对齐 BaseProvider）----

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        if images:
            raise ValueError("当前仅支持文本输入，本地模型不支持图片参数")

        timeout = kwargs.get("timeout", 600)
        try:
            resp = self._post(self._payload(prompt, stream=False, **kwargs), timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
        except requests.RequestException as e:
            raise RuntimeError(f"GGUF 生成失败：{e}") from e

        text = self._strip_think(text)
        # output 回填由外层 executor 的持有式 update_observation 完成
        # （provider 层拿不到 obs，且 update_current_* 依赖的 OTEL context 在
        # iterate_in_threadpool 线程池迭代下不可靠）
        return text

    def generate_stream(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """流式生成：think 段落以 thought 事件下发，正文以 token 事件下发"""
        stop_event = threading.Event()
        with self._stop_lock:
            self._stop_events.add(stop_event)

        if images:
            with self._stop_lock:
                self._stop_events.discard(stop_event)
            yield {"type": "error", "content": "本地模型暂不支持图片输入"}
            return

        with start_span(
            "generate-response",
            as_type="generation",
            input=prompt[:2000],
            metadata={"model": self._model_name, "device": "cpu",
                      "provider": "local-gguf", "stream": True},
        ) as obs:
            processor = TokenProcessor(parse_think=kwargs.get("parse_think", True))
            usage_map: Optional[Dict[str, int]] = None
            stop_patterns = [
                "\n【简历评分】",
                "\n### 【简历评分】",
                "\n## 【简历评分】",
            ]

            token_count = 0
            start_time = time.time()
            try:
                timeout = kwargs.get("timeout", 600)
                with self._post(self._payload(prompt, stream=True, **kwargs), timeout=timeout) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if stop_event.is_set():
                            logger.info("用户请求停止生成（断开连接，服务端随之取消）")
                            yield {"type": "thought", "content": "用户已中止生成"}
                            break
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            if chunk.get("usage"):
                                usage_map = {
                                    "input": chunk["usage"].get("prompt_tokens", 0) or 0,
                                    "output": chunk["usage"].get("completion_tokens", 0) or 0,
                                }
                            continue
                        piece = choices[0].get("delta", {}).get("content")
                        if not piece:
                            continue

                        token_count += 1
                        if token_count == 1:
                            logger.info(
                                f"收到第 1 个 token (首 token 耗时：{time.time() - start_time:.2f}s)"
                            )

                        for output in processor.process_token(piece):
                            yield output

                        if check_stop_patterns(processor.full_text, stop_patterns):
                            logger.info("检测到停止模式，终止生成")
                            break
            except requests.RequestException as e:
                logger.error(f"流式生成错误：{e}")
                yield {"type": "error", "content": f"生成错误：{str(e)}"}
            finally:
                with self._stop_lock:
                    self._stop_events.discard(stop_event)
                if usage_map is not None:
                    update_observation(obs, output=processor.full_text[:2000],
                                       usage=usage_map, metadata={"token_count": token_count})
                else:
                    update_observation(obs, output=processor.full_text[:2000],
                                       metadata={"token_count": token_count})
                logger.info(f"流式生成完成，共 {token_count} 个 token")

    def generate_with_thoughts(self, prompt: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        yield from self.generate_stream(prompt, **kwargs)

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        raise NotImplementedError("本地模型暂不支持图片输入")
