"""
Text Tools - 文本处理工具集

包含：
- TextPolishTool: 文本润色工具
- TextImproverTool: 可读性提升工具

设计原则：
- 支持多种风格转换（专业、简洁、说服、学术）
- 提供详细的修改对比
- 可独立使用或作为 Agent 的一部分
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from ..core.base_tool import BaseTool, ToolResult, ToolRegistry
from ..utils.json_utils import JsonUtils
from ..utils.prompts import TextPrompts, PromptManager


@ToolRegistry.register
class TextPolishTool(BaseTool):
    """
    文本润色工具
    
    功能：
    - 多种风格润色（专业/简洁/说服/学术）
    - 提供修改前后对比
    - 计算可读性评分变化
    
    使用示例：
        tool = TextPolishTool(provider)
        result = tool.execute(
            text="我负责开发了一个系统",
            style="professional"
        )
        print(result.data["polished_text"])
    """
    
    name = "text_polisher"
    description = "润色和优化文本表达，支持专业、简洁、说服力、学术等多种风格，提供详细修改说明"
    
    SUPPORTED_STYLES = ["professional", "concise", "persuasive", "academic"]
    
    def execute(
        self, 
        text: str = "", 
        style: str = "professional",
        **kwargs
    ) -> ToolResult:
        """
        执行文本润色
        
        Args:
            text: 原始文本
            style: 目标风格
            
        Returns:
            ToolResult 包含润色结果
        """
        if not self._provider:
            return ToolResult(
                success=False,
                error="未提供模型提供者",
                metadata={"tool": self.name}
            )
        
        if not text.strip():
            return ToolResult(
                success=False,
                error="文本内容为空",
                data={"polished_text": "", "changes": [], "style": style},
                metadata={"tool": self.name}
            )
        
        if style not in self.SUPPORTED_STYLES:
            style = "professional"
        
        try:
            fallback = PromptManager.get_fallback("polish")
            fallback["polished_text"] = text
            fallback["style"] = style
            
            result = JsonUtils.safe_generate(
                generate_func=self._provider.generate,
                prompt=TextPrompts.get_polish_prompt(text),
                schema_description=TextPrompts.get_polish_schema(style),
                fallback=fallback,
                temperature=0.3,
                max_new_tokens=1024
            )
            
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                metadata={
                    "tool": self.name,
                    "style": style,
                    "original_length": len(text)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"tool": self.name}
            )


@ToolRegistry.register
class TextImproverTool(BaseTool):
    """
    文本可读性提升工具
    
    功能：
    - 简化复杂句子
    - 去除冗余表达
    - 提升清晰度
    
    这是 TextPolishTool 的便捷别名，固定使用 concise 风格
    """
    
    name = "text_improver"
    description = "提升文本可读性和清晰度，去除冗余表达，使语言更简洁明了"
    
    def execute(self, text: str = "", **kwargs) -> ToolResult:
        """
        执行可读性提升
        
        Args:
            text: 原始文本
            
        Returns:
            ToolResult 包含改进后的文本
        """
        if not self._provider:
            return ToolResult(
                success=False,
                error="未提供模型提供者",
                metadata={"tool": self.name}
            )
        
        # 复用 TextPolishTool 的逻辑，使用 concise 风格
        polish_tool = TextPolishTool(provider=self._provider)
        return polish_tool.execute(text=text, style="concise")
