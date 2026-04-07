"""
Local Provider - 本地模型提供者

职责：
- 使用 transformers 库加载和运行本地模型
- 实现文本生成、流式生成、停止控制等核心接口
- 提供模型生命周期管理

设计原则：
- 继承 BaseProvider，实现统一的提供者接口
- 只负责模型交互，不包含业务逻辑
- 通过属性访问 services 层的业务功能

集成：
- 可被 Tool、Service、Agent 层调用
- 支持 LangChain 自定义 LLM 包装
"""

from __future__ import annotations
from logger import get_logger
import torch
import re
import time
import threading
from ..core.base_provider import BaseProvider
from pathlib import Path
from typing import Optional, List, Dict, Generator, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = get_logger(__name__)

try:
    from transformers import pipeline
    from transformers import BitsAndBytesConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import TextIteratorStreamer
except ImportError:
    pipeline = None
    BitsAndBytesConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None


class LocalProvider(BaseProvider):
    """
    本地模型提供者 - 使用 transformers 库
    """

    def __init__(self, model_name: str = "model_serving"):
        """Initialize local model provider."""
        self._model_name = model_name
        self.local_model_path = None
        self.transformers_pipeline = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._stop_generation = False
        self._streamer = None
        self._generate_thread = None
        
        # 服务实例（延迟初始化）
        self._resume_service = None
        self._text_service = None

        # 优先尝试在本地模型目录加载
        model_path = Path(model_name)
        if not model_path.is_absolute():
            root = Path(__file__).resolve().parents[4]
            local_candidate = root / "ai" / "models" / model_name
            if local_candidate.exists():
                model_path = local_candidate

        if model_path.exists():
            self.local_model_path = str(model_path)
            self._load_model()
        else:
            raise FileNotFoundError(
                f"本地模型路径不存在: {model_path}. 仅支持 ai/models 目录下模型。"
            )

    def _load_model(self):
        """加载本地模型"""
        if pipeline is None:
            raise RuntimeError("transformers 库未安装，请安装 transformers 并重试")

        try:
            logger.info(f"Loading model with {self._device.upper()}...")
            self.transformers_pipeline = pipeline(
                "text-generation",
                model=self.local_model_path,
                tokenizer=self.local_model_path,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            logger.info("Using local transformers model: %s (device: %s)", 
                       self.local_model_path, self._device)
        except Exception as e:
            logger.exception("Failed to initialize transformers local model: %s", 
                           self.local_model_path)
            raise RuntimeError(f"Failed to initialize transformers local model "
                             f"({self.local_model_path}): {e}") from e

    @property
    def model_name(self) -> str:
        """返回当前使用的模型名称"""
        return self._model_name
    
    @property
    def device(self) -> str:
        """返回当前设备信息"""
        return self._device

    @property
    def resume_service(self):
        """获取简历服务实例（懒加载）"""
        if self._resume_service is None:
            from ..services.resume_service import ResumeService
            self._resume_service = ResumeService(self)
        return self._resume_service

    @property
    def text_service(self):
        """获取文本服务实例（懒加载）"""
        if self._text_service is None:
            from ..services.text_service import TextService
            self._text_service = TextService(self)
        return self._text_service

    def stop_generation(self):
        """停止当前正在进行的生成任务"""
        self._stop_generation = True
        
        # 停止 streamer
        if self._streamer:
            try:
                if hasattr(self._streamer, 'stop'):
                    self._streamer.stop()
                    logger.info("Streamer stopped")
                if hasattr(self._streamer, 'end'):
                    self._streamer.end()
            except Exception as e:
                logger.error(f"Error stopping streamer: {e}")
        
        # 强制终止生成线程
        if self._generate_thread and self._generate_thread.is_alive():
            import ctypes
            try:
                # 尝试强制终止线程（仅作为最后手段）
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(self._generate_thread.ident),
                    ctypes.py_object(SystemExit)
                )
                if res > 0:
                    logger.info(f"Generate thread termination signal sent (thread={self._generate_thread.ident})")
            except Exception as e:
                logger.warning(f"Failed to terminate thread: {e}")
        
        logger.info("用户请求停止生成")

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """
        从本地模型生成文本
        """
        if images:
            raise ValueError("当前仅支持文本输入，本地模型不支持图片参数")

        if self.transformers_pipeline is None:
            raise RuntimeError("本地模型尚未加载，无法执行生成。")

        timeout = kwargs.get('timeout', 300)

        def _generate_task(prompt, **gen_kwargs):
            tokenizer = self.transformers_pipeline.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt")

            generate_kwargs = {
                **inputs,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": gen_kwargs.get("temperature", 0.3),
                "max_new_tokens": gen_kwargs.get("max_new_tokens", 1024),
                "repetition_penalty": 1.2,
                "top_p": 0.9,
                "top_k": 40,
                "do_sample": True,
            }

            outputs = self.transformers_pipeline.model.generate(**generate_kwargs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_generate_task, prompt, **kwargs)
                result = future.result(timeout=timeout)
                return result
        except TimeoutError:
            raise RuntimeError(f"模型推理超时，超过了 {timeout} 秒的限制")
        except Exception as e:
            raise RuntimeError(f"Local transformers 生成失败: {e}") from e

    def _filter_think_content(self, content: str) -> str:
        """过滤思考内容中的 Prompt 指令重复"""
        patterns = [
            r'按以下\d*个部分输出',
            r'每部分用【标题】开头',
            r'Each part must start with',
            r'Output according to',
            r'indicat.*prompt',
            r'marker.*be formatted',
            r'looking closely at the instruction',
            r'The instruction says',
            r'【简历评分】.*?只给分数',
            r'【优化建议】.*?简短列出',
            r'【优化结果】.*?重写一版',
            r'先用.*结束',
            r'之后按以下',
        ]
        
        lines = content.split('\n')
        filtered = []
        for line in lines:
            should_skip = False
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    should_skip = True
                    break
            if not should_skip and line.strip():
                filtered.append(line)
        
        return '\n'.join(filtered)

    def _process_token(
        self, 
        token: str, 
        in_think_tag: bool, 
        think_buffer: str, 
        full_text: str
    ) -> tuple[bool, str, str, list[Dict[str, Any]]]:
        """处理单个 token，解析 think 标签"""
        outputs = []
        
        # 检测 <think> 开始标签
        if "<think>" in token:
            parts = token.split("<think>", 1)
            
            # <think> 标签之前的内容作为普通文本
            before = parts[0]
            if before.strip():
                full_text += before
                outputs.append({"type": "token", "content": before, "full_text": full_text})
            
            # 检查同一 token 中是否有 </think>
            remaining = parts[1] if len(parts) > 1 else ""
            if "</think>" in remaining:
                # 同一个 token 中有完整的 think 标签
                think_parts = remaining.split("</think>", 1)
                if think_parts[0].strip():
                    filtered_content = self._filter_think_content(think_parts[0].strip())
                    if filtered_content:
                        outputs.append({"type": "thought", "content": filtered_content})
                after = think_parts[1] if len(think_parts) > 1 else ""
                if after.strip():
                    full_text += after
                    outputs.append({"type": "token", "content": after, "full_text": full_text})
            else:
                # 进入 think 模式，保存剩余内容到 buffer
                in_think_tag = True
                if remaining:
                    think_buffer += remaining
                    
        elif "</think>" in token and in_think_tag:
            # 检测 </think> 结束标签
            parts = token.split("</think>", 1)
            think_buffer += parts[0]
            
            if think_buffer.strip():
                filtered_content = self._filter_think_content(think_buffer.strip())
                if filtered_content:
                    outputs.append({"type": "thought", "content": filtered_content})
            
            think_buffer = ""
            in_think_tag = False
            
            # </think> 之后的内容作为普通文本
            after = parts[1] if len(parts) > 1 else ""
            if after.strip():
                full_text += after
                outputs.append({"type": "token", "content": after, "full_text": full_text})
                
        elif in_think_tag:
            # 在 think 标签内，累积内容
            think_buffer += token
            # 遇到换行或积累足够内容时输出（避免逐 token 碎片化显示）
            # 但要检查是否包含 </think> 结束标签
            if '\n' in token or len(think_buffer) >= 30:
                # 先检查 buffer 中是否有 </think>
                if "</think>" in think_buffer:
                    # buffer 中有结束标签，需要分割处理
                    think_parts = think_buffer.split("</think>", 1)
                    if think_parts[0].strip():
                        filtered_content = self._filter_think_content(think_parts[0].strip())
                        if filtered_content:
                            outputs.append({"type": "thought", "content": filtered_content})
                    
                    think_buffer = ""
                    in_think_tag = False
                    
                    # </think> 之后的内容作为普通文本
                    if len(think_parts) > 1 and think_parts[1].strip():
                        full_text += think_parts[1]
                        outputs.append({"type": "token", "content": think_parts[1], "full_text": full_text})
                else:
                    # 没有结束标签，正常输出思考内容
                    filtered_content = self._filter_think_content(think_buffer)
                    if filtered_content:
                        outputs.append({"type": "thought", "content": filtered_content})
                    think_buffer = ""
        else:
            # 普通文本（不在 think 标签内）
            full_text += token
            outputs.append({"type": "token", "content": token, "full_text": full_text})
        
        return in_think_tag, think_buffer, full_text, outputs

    def generate_with_thoughts(
        self, 
        prompt: str, 
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成文本，支持停止生成和 think 标签解析
        """
        self._stop_generation = False
        
        full_text = ""
        in_think_tag = False
        think_buffer = ""
        
        if self.transformers_pipeline is None:
            yield {"type": "error", "content": "本地模型尚未加载，无法执行生成。"}
            return
        
        if TextIteratorStreamer is None:
            yield {"type": "error", "content": "transformers 库版本过低，不支持流式输出"}
            return
        
        timeout = kwargs.get('timeout', 300)
        start_time = time.time()
        
        try:
            tokenizer = self.transformers_pipeline.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt")
            
            streamer = TextIteratorStreamer(
                tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True,
                timeout=timeout,
                chunk_size=1
            )
            
            self._streamer = streamer
            
            generate_kwargs = {
                **inputs,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": kwargs.get("temperature", 0.3),
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
                "repetition_penalty": 1.15,
                "top_p": 0.85,
                "top_k": 30,
                "do_sample": True,
            }

            thread = threading.Thread(
                target=self.transformers_pipeline.model.generate,
                kwargs=generate_kwargs,
                daemon=True
            )
            self._generate_thread = thread
            thread.start()
            
            last_token = ""
            stop_patterns = [
                "\n【简历评分】",  # 第二次出现简历评分，说明开始重复
                "\n### 【简历评分】",
                "\n## 【简历评分】",
            ]
            
            try:
                for token in streamer:
                    if self._stop_generation:
                        yield {"type": "thought", "content": "用户已中止生成"}
                        if hasattr(streamer, 'stop'):
                            streamer.stop()
                        return
                    
                    if time.time() - start_time > timeout:
                        if not self._stop_generation:
                            yield {"type": "thought", 
                                  "content": f"模型推理超时（超过 {timeout} 秒），已生成部分内容"}
                            # 同时输出已生成的内容到优化建议
                            partial_text = re.sub(r'', '', full_text)
                            if partial_text.strip():
                                yield {"type": "token", "content": partial_text}
                        return
                    
                    token_clean = token.strip()
                    if not token_clean or token_clean == last_token:
                        continue
                    
                    last_token = token_clean
                    
                    # 检测是否需要停止（避免重复输出）
                    full_text_lower = full_text.lower()
                    if "【优化结果】" in full_text:
                        # 已经输出过优化结果，检查是否开始重复
                        for pattern in stop_patterns:
                            if pattern in full_text:
                                logger.info(f"检测到重复模式 '{pattern}'，停止生成")
                                return
                    
                    in_think_tag, think_buffer, full_text, outputs = self._process_token(
                        token, in_think_tag, think_buffer, full_text
                    )
                    
                    if self._stop_generation:
                        yield {"type": "thought", "content": "用户已中止生成"}
                        if hasattr(streamer, 'stop'):
                            streamer.stop()
                        return
                    
                    for output in outputs:
                        yield output
                        
            except Exception as e:
                if any(keyword in str(e).lower() 
                      for keyword in ["stop", "abort", "canceled"]):
                    logger.info("Generation stopped by user")
                    yield {"type": "thought", "content": "用户已中止生成"}
                    if hasattr(streamer, 'stop'):
                        streamer.stop()
                else:
                    logger.error(f"Error during generation: {e}")
                    
            if self._stop_generation:
                yield {"type": "thought", "content": "用户已中止生成"}
                return
            
            thread.join(timeout=5)
            
            if self._stop_generation:
                yield {"type": "thought", "content": "用户已中止生成"}
                return
            
            if think_buffer.strip():
                yield {"type": "thought", "content": think_buffer.strip()}
            
            full_text = re.sub(r'', '', full_text)
            
            # 清理所有位置的角色前缀（开头、中间、结尾）
            prefix_patterns = [
                r'\s*assistant\s*', 
                r'\s*Assistant\s*',
                r'\s*AI助手\s*',
                r'\s*回复[：:]\s*',
                r'\s*回答[：:]\s*',
            ]
            
            cleaned = full_text
            for pattern in prefix_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # 清理多余空白
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
            
            yield {"type": "complete", "full_text": cleaned}
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            if any(keyword in str(e).lower() 
                  for keyword in ["abort", "stop"]):
                yield {"type": "thought", "content": "用户已中止生成"}
            else:
                yield {"type": "error", "content": f"Local transformers 生成失败: {str(e)}"}
        finally:
            self._stop_generation = True
            self._streamer = None

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """当前本地模型仅支持文本推理。"""
        raise NotImplementedError("本地模型当前仅支持文本输入，暂不支持图像生成。")


# 向后兼容：保留 LocalAgent 作为 LocalProvider 的别名
LocalAgent = LocalProvider
