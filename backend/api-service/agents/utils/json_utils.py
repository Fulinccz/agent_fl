"""
JSON Utility Functions
"""

from __future__ import annotations
import re
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class JsonUtils:
    """JSON 处理工具类"""
    
    @staticmethod
    def extract_json(response: str) -> Optional[Dict[str, Any]]:
        """
        从模型响应中提取并解析 JSON
        
        Args:
            response: 模型返回的原始文本
            
        Returns:
            解析后的字典，如果解析失败返回 None
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                json_str = JsonUtils._clean_repeated_content(json_str)
                return json.loads(json_str)
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"JSON 解析失败: {e}")
            return None
    
    @staticmethod
    def _clean_repeated_content(text: str) -> str:
        """清理文本中的重复内容"""
        return re.sub(r'([^\n]+)\1+', r'\1', text)
    
    @staticmethod
    def validate_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        验证 JSON 数据是否包含必需字段
        
        Args:
            data: 要验证的字典数据
            required_keys: 必需的字段列表
            
        Returns:
            如果包含所有必需字段返回 True，否则 False
        """
        return all(key in data for key in required_keys)
    
    @staticmethod
    def safe_generate(
        generate_func,
        prompt: str,
        schema_description: str,
        fallback: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        安全地生成 JSON 格式的响应
        
        Args:
            generate_func: 生成函数（如 agent.generate）
            prompt: 用户输入的提示词
            schema_description: JSON Schema 描述
            fallback: 解析失败时的默认返回值
            **kwargs: 传递给 generate_func 的额外参数
            
        Returns:
            成功：解析后的 JSON 字典
            失败：fallback 字典（包含错误信息）
        """
        full_prompt = f"""{schema_description}

{prompt}
"""
        
        try:
            response = generate_func(full_prompt, **kwargs)
            result = JsonUtils.extract_json(response)
            
            if result and JsonUtils.validate_structure(result, list(fallback.keys())):
                return result
            
            logger.warning("JSON 解析失败或结构不完整，使用 fallback")
            return fallback
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            fallback["error"] = str(e)
            return fallback
    
    @staticmethod
    def create_error_response(
        task_type: str,
        error_message: str,
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        创建标准化的错误响应
        
        Args:
            task_type: 任务类型（如 'parse', 'optimize', 'score'）
            error_message: 错误信息
            additional_data: 额外的错误数据
            
        Returns:
            标准化的错误响应字典
        """
        response = {
            "success": False,
            "task_type": task_type,
            "error": error_message
        }
        
        if additional_data:
            response.update(additional_data)
            
        return response
    
    @staticmethod
    def create_success_response(
        task_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建标准化的成功响应
        
        Args:
            task_type: 任务类型
            data: 响应数据
            
        Returns:
            标准化的成功响应字典
        """
        return {
            "success": True,
            "task_type": task_type,
            "error": None,
            **data
        }
