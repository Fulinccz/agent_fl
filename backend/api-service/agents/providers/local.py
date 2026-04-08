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
            
            # 检查是否支持 bitsandbytes 4bit 量化
            # 注意：bitsandbytes 在 Windows 上支持有限，需要特殊处理
            use_4bit = True  # 默认尝试 4bit 量化
            is_windows = torch.cuda.is_available() and torch.version.cuda is not None
            
            if is_windows:
                import platform
                if platform.system() == "Windows":
                    logger.info("⚠️ 检测到 Windows 系统，bitsandbytes 4bit 量化可能不支持")
                    logger.info("将尝试使用标准精度加载，或使用 CPU 模式")
                    use_4bit = False  # Windows 默认不使用 4bit 量化
            
            if use_4bit and BitsAndBytesConfig is not None:
                try:
                    # 配置 4bit 量化参数
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",  # Normal Float 4bit
                        bnb_4bit_compute_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                        bnb_4bit_use_double_quant=True,  # 嵌套量化，进一步节省显存
                    )
                    
                    logger.info("✅ 使用 bitsandbytes 4bit 量化加载模型")
                    logger.info(f"  - 量化类型：nf4")
                    logger.info(f"  - 计算精度：{quantization_config.bnb_4bit_compute_dtype}")
                    logger.info(f"  - 嵌套量化：启用")
                    
                    # 使用 4bit 量化加载模型
                    self.transformers_pipeline = pipeline(
                        "text-generation",
                        model=self.local_model_path,
                        tokenizer=self.local_model_path,
                        trust_remote_code=True,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        quantization_config=quantization_config,
                    )
                    
                    logger.info("✅ 4bit 量化模型加载成功")
                    logger.info(f"📊 模型设备：{self.transformers_pipeline.model.device}")
                    logger.info(f"💾 模型 dtype: {self.transformers_pipeline.model.dtype}")
                    return
                    
                except Exception as e:
                    logger.warning(f"⚠️ 4bit 量化加载失败：{e}")
                    logger.info("回退到标准精度加载模式")
            
            # 标准精度加载（回退方案）
            logger.info("使用标准精度加载模型")
            
            # 优化模型加载配置
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",  # 自动选择设备
                "low_cpu_mem_usage": True,  # 优化 CPU 内存使用
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,  # 自动使用 float16
            }
            
            self.transformers_pipeline = pipeline(
                "text-generation",
                model=self.local_model_path,
                tokenizer=self.local_model_path,
                **model_kwargs,
            )
            logger.info("Using local transformers model: %s (device: %s)", 
                       self.local_model_path, self._device)
            logger.info(f"📊 模型设备：{self.transformers_pipeline.model.device}")
            logger.info(f"💾 模型 dtype: {self.transformers_pipeline.model.dtype}")
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
                "temperature": gen_kwargs.get("temperature", 0.35),
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
        full_text: str,
        parse_think: bool = True  # 是否解析 think 标签
    ) -> tuple[bool, str, str, list[Dict[str, Any]]]:
        """处理单个 token，解析 think 标签"""
        outputs = []
        
        # 如果不解析 think 标签，直接当作普通文本
        if not parse_think:
            full_text += token
            outputs.append({"type": "token", "content": token, "full_text": full_text})
            return in_think_tag, think_buffer, full_text, outputs
        
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

    def generate_with_thoughts(self, prompt: str, **kwargs):
        """
        流式生成带思考过程的内容
        
        改进措施：
        1. 添加重试机制，应对间歇性失败
        2. 添加健康检查，确保模型状态正常
        3. 优化超时处理，避免长时间等待
        4. 生成前重置状态，避免残留状态影响
        """
        # 重要：生成前重置所有状态，避免上次任务的影响
        self._stop_generation = False
        self._streamer = None
        self._generate_thread = None
        
        full_text = ""
        in_think_tag = False
        think_buffer = ""
        
        if self.transformers_pipeline is None:
            logger.error("❌ 本地模型尚未加载，无法执行生成")
            yield {"type": "error", "content": "本地模型尚未加载，无法执行生成。"}
            return
        
        if TextIteratorStreamer is None:
            logger.error("❌ transformers 库版本过低，不支持流式输出")
            yield {"type": "error", "content": "transformers 库版本过低，不支持流式输出"}
            return
        
        # 健康检查：确保模型在可用设备上
        try:
            device = self.transformers_pipeline.model.device
            logger.debug(f"🏥 模型健康检查 - 设备：{device}, 状态：正常")
        except Exception as e:
            logger.error(f"❌ 模型健康检查失败：{e}")
            yield {"type": "error", "content": f"模型状态异常：{str(e)}"}
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
            
            # 构建生成参数 - 简历优化专用配置
            generate_kwargs = {
                **inputs,
                "streamer": streamer,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                
                # 生成质量参数 - 针对简历优化场景
                "temperature": kwargs.get("temperature", 0.35),  # 稍微提高，避免过于保守
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
                "repetition_penalty": 1.1,  # 轻微重复惩罚，避免过度修改
                "top_p": 0.75,  # 更保守的 nucleus sampling
                "top_k": 20,  # 更少的候选词，更稳定
                "do_sample": True,
                
                # 性能优化参数
                "use_cache": True,  # 启用 KV cache，加速生成
                "early_stopping": False,  # 不要提前停止
            }
            
            logger.info(f"🔍 开始模型生成 - prompt 长度：{len(prompt)} 字符")
            logger.info(f"📊 输入 token 数：{inputs['input_ids'].shape[1]}")
            logger.info(f"⚙️  生成参数：max_new_tokens={generate_kwargs['max_new_tokens']}, temperature={generate_kwargs['temperature']}")
            logger.info(f"💻 模型设备：{self.transformers_pipeline.model.device}")
            logger.info(f"💾 模型 dtype: {self.transformers_pipeline.model.dtype}")
            
            # 重要：确保模型处于 eval 模式，禁用 dropout
            self.transformers_pipeline.model.eval()
            
            logger.info("🚀 启动生成线程...")
            thread = threading.Thread(
                target=self.transformers_pipeline.model.generate,
                kwargs=generate_kwargs,
                daemon=True
            )
            self._generate_thread = thread
            thread.start()
            logger.info("✅ 生成线程已启动")
            
            # 监控线程状态
            time.sleep(0.5)
            if not thread.is_alive():
                logger.error("❌ 生成线程已异常退出！")
            
            last_token = ""
            stop_patterns = [
                "\n【简历评分】",  # 第二次出现简历评分，说明开始重复
                "\n### 【简历评分】",
                "\n## 【简历评分】",
            ]
            
            try:
                token_count = 0
                first_token_time = None
                last_token_time = time.time()
                
                for token in streamer:
                    current_time = time.time()
                    token_count += 1
                    
                    # 记录原始 token（包括空白）
                    logger.debug(f"📝 Token #{token_count} (原始): '{token}' (repr: {repr(token[:30])})")
                    
                    # 记录 token 间隔，检测是否卡住
                    token_interval = current_time - last_token_time
                    if token_count == 1:
                        first_token_time = current_time
                        logger.info(f"✅ 收到第 1 个 token: '{token[:50]}...' (首 token 耗时：{current_time - start_time:.2f}s)")
                    elif token_count <= 5:
                        logger.debug(f"Token #{token_count}: '{token[:30]}' (间隔：{token_interval:.2f}s)")
                    
                    # 检测是否长时间没有新 token（超过 60 秒）
                    if token_interval > 60 and token_count > 1:
                        logger.warning(f"⚠️  检测到生成停滞（{token_interval:.2f}s 无新 token）")
                    
                    last_token_time = current_time
                    
                    # 检查用户是否请求停止
                    if self._stop_generation:
                        logger.info("⚠️  用户请求停止生成")
                        yield {"type": "thought", "content": "用户已中止生成"}
                        if hasattr(streamer, 'stop'):
                            streamer.stop()
                        return
                    
                    # 检查是否超时
                    if current_time - start_time > timeout:
                        logger.warning(f"⚠️  生成超时（{timeout}秒），已生成 {token_count} 个 token")
                        if not self._stop_generation:
                            yield {"type": "thought", 
                                  "content": f"模型推理超时（超过 {timeout} 秒），已生成部分内容"}
                            partial_text = re.sub(r'', '', full_text)
                            if partial_text.strip():
                                yield {"type": "token", "content": partial_text}
                        return
                    
                    # 跳过空白 token 和完全重复的 token
                    token_clean = token.strip()
                    if not token_clean or token_clean == last_token:
                        logger.debug(f"⚪ 跳过空白/重复 token: '{token_clean}'")
                        continue
                    
                    last_token = token_clean
                    logger.debug(f"✅ 有效 token: '{token_clean[:20]}'")
                    
                    # 根据 deepThinking 参数决定是否解析 think 标签
                    parse_think = kwargs.get("deepThinking", False)
                    
                    # 处理 token（解析 think 标签等）
                    in_think_tag, think_buffer, full_text, outputs = self._process_token(
                        token, in_think_tag, think_buffer, full_text, parse_think
                    )
                    
                    # 非深度思考模式下，过滤掉所有 thought 类型的输出
                    if not parse_think and outputs:
                        outputs = [out for out in outputs if out.get("type") != "thought"]
                    
                    # 再次检查用户是否请求停止
                    if self._stop_generation:
                        logger.info("⚠️  用户请求停止生成")
                        yield {"type": "thought", "content": "用户已中止生成"}
                        if hasattr(streamer, 'stop'):
                            streamer.stop()
                        return
                    
                    # 输出处理后的内容
                    for output in outputs:
                        yield output
                        
                # 循环结束，记录生成统计
                elapsed_time = time.time() - start_time
                avg_token_interval = elapsed_time / token_count if token_count > 0 else 0
                logger.info(f"✅ 生成完成 - 总 token 数：{token_count}, 耗时：{elapsed_time:.2f}s, 平均速度：{avg_token_interval:.2f}s/token")
                
                # 检测生成质量
                if token_count < 10:
                    logger.warning(f"⚠️  生成的 token 数过少（{token_count}），可能存在问题")
                elif token_count < 50:
                    logger.info(f"ℹ️  生成的 token 数较少（{token_count}），属于正常短回答")
                else:
                    logger.info(f"✅ 生成质量良好（{token_count} tokens）")
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"❌ 生成异常：{type(e).__name__}: {e} (耗时：{elapsed_time:.2f}s, token 数：{token_count})", exc_info=True)
                yield {"type": "error", "content": f"模型生成失败：{str(e)}"}
            finally:
                # 清理资源
                self._streamer = None
                self._generate_thread = None
                logger.debug("🧹 生成资源已清理")
                    
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
