from __future__ import annotations
from logger import get_logger
import torch
from .base import BaseAgent
from pathlib import Path
from typing import Optional, List, Dict, Generator, Any
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re
import json

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


class LocalAgent(BaseAgent):
    """A local-only model adapter using transformers from ai/models."""

    def __init__(self, model_name: str = "model_serving"):
        """Initialize local model adapter."""
        self.model_name = model_name
        self.local_model_path = None
        self.transformers_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 优先尝试在本地模型目录加载
        model_path = Path(model_name)
        if not model_path.is_absolute():
            root = Path(__file__).resolve().parents[3]
            local_candidate = root / "ai" / "models" / model_name
            if local_candidate.exists():
                model_path = local_candidate

        if model_path.exists():
            self.local_model_path = str(model_path)
            if pipeline is None:
                raise RuntimeError("transformers 库未安装，请安装 transformers 并重试")

            try:
                logger.info(f"Loading model with {self.device.upper()}...")
                self.transformers_pipeline = pipeline(
                    "text-generation",
                    model=self.local_model_path,
                    tokenizer=self.local_model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                logger.info("Using local transformers model: %s (device: %s)", self.local_model_path, self.device)
            except Exception as e:
                logger.exception("Failed to initialize transformers local model: %s", self.local_model_path)
                raise RuntimeError(f"Failed to initialize transformers local model ({self.local_model_path}): {e}") from e
        else:
            raise FileNotFoundError(
                f"本地模型路径不存在: {model_path}. 仅支持 ai/models 目录下模型。"
            )
        
        # 初始化停止标志
        self._stop_generation = False

    def stop_generation(self):
        """停止生成"""
        self._stop_generation = True
        # 尝试停止streamer
        if hasattr(self, '_streamer') and self._streamer:
            try:
                if hasattr(self._streamer, 'stop'):
                    self._streamer.stop()
                    logger.info("Streamer stopped")
            except Exception as e:
                logger.error(f"Error stopping streamer: {e}")
        logger.info("用户请求停止生成")

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text from local transformers model only."""
        if images:
            raise ValueError("当前仅支持文本输入，本地模型不支持图片参数")

        if self.transformers_pipeline is None:
            raise RuntimeError("本地模型尚未加载，无法执行生成。")

        timeout = kwargs.get('timeout', 300)

        def generate_task(self, prompt, **kwargs):
            """封装生成任务，用于线程执行"""
            tokenizer = self.transformers_pipeline.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt")

            # ========== 核心修复1：添加去重参数，解决无限重复 ==========
            generate_kwargs = {
                **inputs,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": kwargs.get("temperature", 0.3),  # 降低温度，减少发散
                "max_new_tokens": kwargs.get("max_new_tokens", 1024),
                "repetition_penalty": 1.2,  # 关键：解决重复问题
                "top_p": 0.9,
                "top_k": 40,
                "do_sample": True,
            }

            outputs = self.transformers_pipeline.model.generate(**generate_kwargs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_task, self, prompt,** kwargs)
                result = future.result(timeout=timeout)
                return result
        except TimeoutError:
            raise RuntimeError(f"模型推理超时，超过了 {timeout} 秒的限制")
        except Exception as e:
            raise RuntimeError(f"Local transformers 生成失败: {e}") from e

    def _process_token(self, token: str, in_think_tag: bool, think_buffer: str, full_text: str) -> tuple[bool, str, str, list[Dict[str, Any]]]:
        """
        处理单个token，解析think标签并返回处理结果
        
        Args:
            token: 输入token
            in_think_tag: 是否在think标签内
            think_buffer: 当前think内容缓冲区
            full_text: 当前完整文本
            
        Returns:
            (新的in_think_tag状态, 新的think_buffer, 新的full_text, 要yield的内容列表)
        """
        outputs = []
        
        # 解析think标签
        if "<think>" in token and not in_think_tag:
            in_think_tag = True
            # 提取<think>之后的内容
            think_start = token.find("<think>") + len("<think>")
            if think_start < len(token):
                think_buffer += token[think_start:]
        elif "</think>" in token and in_think_tag:
            in_think_tag = False
            # 提取</think>之前的内容
            think_end = token.find("</think>")
            think_buffer += token[:think_end]
            # 发送think内容
            if think_buffer.strip():
                outputs.append({"type": "thought", "content": think_buffer.strip()})
            think_buffer = ""
            # 提取</think>之后的内容作为正常输出
            after_think = token[think_end + len("</think>"):]
            if after_think:
                full_text += after_think
                outputs.append({"type": "token", "content": after_think, "full_text": full_text})
        elif in_think_tag:
            think_buffer += token
        else:
            # 正常输出（不在think标签内）
            full_text += token
            outputs.append({"type": "token", "content": token, "full_text": full_text})
        
        return in_think_tag, think_buffer, full_text, outputs

    def generate_with_thoughts(self, prompt: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成文本，支持停止生成和think标签解析
        """
        # 重置停止标志
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
            
            # 流式生成参数
            streamer = TextIteratorStreamer(
                tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True,
                timeout=timeout,
                chunk_size=1
            )
            
            # 保存streamer引用，用于停止生成
            self._streamer = streamer
            
            generate_kwargs = {
                **inputs,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": kwargs.get("temperature", 0.3),
                "max_new_tokens": kwargs.get("max_new_tokens", 1024),
                "repetition_penalty": 1.2,
                "top_p": 0.9,
                "top_k": 40,
                "do_sample": True,
            }

            import threading
            thread = threading.Thread(
                target=self.transformers_pipeline.model.generate,
                kwargs=generate_kwargs,
                daemon=True
            )
            thread.start()
            
            # 防止重复token累积
            last_token = ""
            
            try:
                for token in streamer:
                    # 检查是否停止生成
                    if self._stop_generation:
                        yield {"type": "thought", "content": "用户已中止生成"}
                        # 尝试停止streamer
                        if hasattr(streamer, 'stop'):
                            streamer.stop()
                        return
                    
                    if time.time() - start_time > timeout:
                        # 超时前再次检查是否已停止
                        if not self._stop_generation:
                            yield {"type": "thought", "content": f"模型推理超时（超过 {timeout} 秒），已生成部分内容"}
                        return
                    
                    # 过滤空token + 重复token
                    token_clean = token.strip()
                    if not token_clean or token_clean == last_token:
                        continue
                    
                    # 更新最后一个token
                    last_token = token_clean
                    
                    # 处理token，解析think标签
                    in_think_tag, think_buffer, full_text, outputs = self._process_token(
                        token, in_think_tag, think_buffer, full_text
                    )
                    
                    # 检查是否在处理过程中被停止
                    if self._stop_generation:
                        yield {"type": "thought", "content": "用户已中止生成"}
                        # 尝试停止streamer
                        if hasattr(streamer, 'stop'):
                            streamer.stop()
                        return
                    
                    # 输出处理结果
                    for output in outputs:
                        yield output
            except Exception as e:
                # 捕获可能的异常，如streamer被中断
                if "stop" in str(e).lower() or "abort" in str(e).lower() or "canceled" in str(e).lower():
                    logger.info("Generation stopped by user")
                    yield {"type": "thought", "content": "用户已中止生成"}
                    # 尝试停止streamer
                    if hasattr(streamer, 'stop'):
                        streamer.stop()
                else:
                    logger.error(f"Error during generation: {e}")
            
            # 检查是否在循环结束后被停止
            if self._stop_generation:
                yield {"type": "thought", "content": "用户已中止生成"}
                return
            
            thread.join(timeout=5)
            
            # 检查是否在线程等待后被停止
            if self._stop_generation:
                yield {"type": "thought", "content": "用户已中止生成"}
                return
            
            # 如果还有未发送的think内容，发送它
            if think_buffer.strip():
                yield {"type": "thought", "content": think_buffer.strip()}
            
            # 生成完成，确保full_text中不包含think标签
            full_text = re.sub(r'<think>[\s\S]*?</think>', '', full_text)
            yield {"type": "complete", "full_text": full_text}
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            # 检查是否是中止错误
            if "abort" in str(e).lower() or "stop" in str(e).lower():
                yield {"type": "thought", "content": "用户已中止生成"}
            else:
                yield {"type": "error", "content": f"Local transformers 生成失败: {str(e)}"}
        finally:
            # 确保停止标志被设置
            self._stop_generation = True
            # 清理streamer引用
            if hasattr(self, '_streamer'):
                delattr(self, '_streamer')

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """当前本地模型仅支持文本推理。"""
        raise NotImplementedError("本地模型当前仅支持文本输入，暂不支持图像生成。")
    
    def optimize_resume(self, resume_content: str, job_description: str = "") -> Dict[str, Any]:
        """
        优化简历内容，返回结构化的优化结果
        """
        if not resume_content.strip():
            return {
                "resume_score": 0,
                "strengths": [],
                "weaknesses": ["简历内容为空"],
                "improved_paragraph": "",
                "matched_skills_from_JD": [],
                "final_suggestion": "简历内容为空，请提供有效的简历文本。"
            }

        # 简化prompt，避免模型压力过大导致重复
        prompt = f"""你是专业简历优化师，严格按照以下JSON格式输出，不输出任何多余内容：
{{
    "resume_score": 0~100的整数,
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["劣势1", "劣势2"],
    "improved_paragraph": "优化后的简历段落，禁止重复内容",
    "matched_skills_from_JD": ["技能1", "技能2"],
    "final_suggestion": "一句话优化建议"
}}

简历内容：{resume_content}
{"职位描述：" + job_description if job_description else ""}

要求：improved_paragraph 部分禁止重复相同模块，语言简洁有条理。"""
        
        try:
            response = self.generate(prompt, temperature=0.3, max_new_tokens=1024)
            
            # 增强JSON提取，兼容多行
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                # 清理重复内容
                json_str = re.sub(r'([^\n]+)\1+', r'\1', json_str)
                result = json.loads(json_str)
                # 额外清理improved_paragraph的重复内容
                if "improved_paragraph" in result:
                    result["improved_paragraph"] = re.sub(r'([^\n。；]+)\1+', r'\1', result["improved_paragraph"])
                return result
            else:
                logger.warning(f"未提取到JSON，原始响应: {response[:500]}")
                return {
                    "resume_score": 50,
                    "strengths": [],
                    "weaknesses": [],
                    "improved_paragraph": resume_content,
                    "matched_skills_from_JD": [],
                    "final_suggestion": "无法解析模型输出，请检查输入内容。"
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始内容: {response[:500]}")
            return {
                "resume_score": 50,
                "strengths": [],
                "weaknesses": [],
                "improved_paragraph": resume_content,
                "matched_skills_from_JD": [],
                "final_suggestion": f"优化失败: {str(e)}"
            }
        except Exception as e:
            logger.error(f"简历优化失败: {e}")
            return {
                "resume_score": 50,
                "strengths": [],
                "weaknesses": [],
                "improved_paragraph": resume_content,
                "matched_skills_from_JD": [],
                "final_suggestion": f"优化失败: {str(e)}"
            }