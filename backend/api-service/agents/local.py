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
                "early_stopping": True,  # 生成完成后停止，避免无限输出
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

    def generate_with_thoughts(self, prompt: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成文本，修复重复+完整显示
        """
        full_text = ""
        yield {"type": "thought", "content": f"收到用户输入: {prompt[:100]}..." if len(prompt) > 100 else f"收到用户输入: {prompt}"}
        
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
            
            # ========== 核心修复2：流式生成也添加去重参数 ==========
            streamer = TextIteratorStreamer(
                tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True,
                timeout=timeout,
                # 修复：每次只返回1个token，避免前端渲染卡顿
                chunk_size=1
            )
            
            generate_kwargs = {
                **inputs,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": kwargs.get("temperature", 0.3),
                "max_new_tokens": kwargs.get("max_new_tokens", 1024),
                "repetition_penalty": 1.2,  # 流式生成也加去重
                "top_p": 0.9,
                "top_k": 40,
                "do_sample": True,
                "early_stopping": True,
            }

            import threading
            thread = threading.Thread(
                target=self.transformers_pipeline.model.generate,
                kwargs=generate_kwargs,
                daemon=True
            )
            thread.start()
            
            yield {"type": "thought", "content": "开始流式生成文本..."}
            
            # ========== 核心修复3：防止重复token累积 ==========
            last_token = ""  # 记录上一个token，避免重复
            for token in streamer:
                if time.time() - start_time > timeout:
                    yield {"type": "thought", "content": f"模型推理超时（超过 {timeout} 秒），已生成部分内容"}
                    return
                
                # 过滤空token + 重复token
                token_clean = token.strip()
                if not token_clean or token_clean == last_token:
                    continue
                
                # 更新最后一个token
                last_token = token_clean
                full_text += token
                
                # ========== 核心修复4：确保前端能完整接收 ==========
                yield {
                    "type": "token", 
                    "content": token,
                    "full_text": full_text,
                    # 新增：标记是否是最后一个token，方便前端处理
                    "is_last": False
                }
            
            thread.join(timeout=5)
            # 生成完成，告诉前端结束
            yield {
                "type": "thought", 
                "content": "推理结束，生成完成。",
                "full_text": full_text,
                "is_last": True
            }
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield {"type": "error", "content": f"Local transformers 生成失败: {str(e)}"}

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