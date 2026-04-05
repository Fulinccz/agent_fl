"""
Text Service
文本处理相关的业务服务（润色、改写等）

职责：
- 文本润色：提升表达质量
- 风格转换：专业/简洁/说服/学术
- 可读性优化

设计原则：
- 单一职责：只处理文本相关业务
- 可扩展：易于添加新的文本处理功能
- 统一输出：标准 JSON 格式
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from ..utils.json_utils import JsonUtils
from ..utils.prompts import TextPrompts, PromptManager


class TextService:
    """文本服务类 - 处理所有文本处理任务"""
    
    SUPPORTED_STYLES = ["professional", "concise", "persuasive", "academic"]
    
    def __init__(self, agent):
        """
        初始化文本服务
        
        Args:
            agent: Agent 实例（LocalAgent 或 OnlineAgent）
        """
        self.agent = agent
        self.prompts = TextPrompts()
    
    def polish(
        self,
        text: str,
        style: str = "professional"
    ) -> Dict[str, Any]:
        """
        润色文本，提升表达质量
        
        Args:
            text: 原始文本内容
            style: 目标风格（professional/concise/persuasive/academic）
            
        Returns:
            包含润色结果的字典：
            {
                "success": bool,
                "original_text": str,
                "polished_text": str,
                "style": str,
                "changes": [...],
                ...
            }
        """
        if not text.strip():
            return JsonUtils.create_error_response(
                task_type="polish",
                error_message="文本内容为空",
                additional_data={
                    "polished_text": "",
                    "changes": [],
                    "style": style
                }
            )
        
        if style not in self.SUPPORTED_STYLES:
            style = "professional"
        
        fallback = PromptManager.get_fallback("polish")
        fallback["polished_text"] = text
        fallback["style"] = style
        
        return JsonUtils.safe_generate(
            generate_func=self.agent.generate,
            prompt=self.prompts.get_polish_prompt(text),
            schema_description=self.prompts.get_polish_schema(style),
            fallback=fallback,
            temperature=0.3,
            max_new_tokens=1024
        )
    
    def improve_readability(self, text: str) -> Dict[str, Any]:
        """专门提升文本可读性（简洁风格的别名）"""
        return self.polish(text, style="concise")
    
    def make_professional(self, text: str) -> Dict[str, Any]:
        """使文本更专业化（专业风格的别名）"""
        return self.polish(text, style="professional")
    
    def make_persuasive(self, text: str) -> Dict[str, Any]:
        """增强文本的说服力"""
        return self.polish(text, style="persuasive")
