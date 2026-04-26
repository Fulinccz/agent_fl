"""
Token 处理器
处理流式生成中的 token，包括 think 标签解析
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .utils import filter_think_content


class TokenProcessor:
    """Token 处理器 - 解析 think 标签和普通 token"""
    
    def __init__(self, parse_think: bool = True):
        self.parse_think = parse_think
        self.in_think_tag = False
        self.think_buffer = ""
        self.full_text = ""
    
    def process_token(self, token: str) -> List[Dict[str, Any]]:
        """
        处理单个 token
        
        Returns:
            包含 type 和 content 的字典列表
        """
        outputs = []
        
        # 如果不解析 think 标签，仍然需要过滤掉 think 标签内容
        if not self.parse_think:
            return self._process_without_think(token)
        
        # 检测 <think> 开始标签
        if "<think>" in token:
            return self._handle_think_start(token)
        
        # 检测 </think> 结束标签
        if "</think>" in token and self.in_think_tag:
            return self._handle_think_end(token)
        
        # 普通 token 处理
        if self.in_think_tag:
            # 在 think 标签内，累积到缓冲区
            self.think_buffer += token
        else:
            # 在 think 标签外，直接输出
            self.full_text += token
            outputs.append({"type": "token", "content": token, "full_text": self.full_text})
        
        return outputs
    
    def _process_without_think(self, token: str) -> List[Dict[str, Any]]:
        """不解析 think 标签，直接过滤掉"""
        outputs = []
        
        # 检测 <think> 开始
        if "<think>" in token and not self.in_think_tag:
            parts = token.split("<think>", 1)
            before = parts[0]
            if before.strip():
                self.full_text += before
                outputs.append({"type": "token", "content": before, "full_text": self.full_text})
            self.in_think_tag = True
            self.think_buffer = parts[1] if len(parts) > 1 else ""
            return outputs
        
        # 检测 </think> 结束
        if "</think>" in token and self.in_think_tag:
            parts = token.split("</think>", 1)
            self.think_buffer += parts[0]
            self.in_think_tag = False
            # 只输出 </think> 之后的内容
            after = parts[1] if len(parts) > 1 else ""
            if after.strip():
                self.full_text += after
                outputs.append({"type": "token", "content": after, "full_text": self.full_text})
            return outputs
        
        # 在 think 标签内，跳过
        if self.in_think_tag:
            self.think_buffer += token
            return outputs
        
        # 普通 token，直接输出
        self.full_text += token
        outputs.append({"type": "token", "content": token, "full_text": self.full_text})
        return outputs
    
    def _handle_think_start(self, token: str) -> List[Dict[str, Any]]:
        """处理 think 标签开始"""
        outputs = []
        parts = token.split("<think>", 1)
        
        # <think> 标签之前的内容作为普通文本
        before = parts[0]
        if before.strip():
            self.full_text += before
            outputs.append({"type": "token", "content": before, "full_text": self.full_text})
        
        # 进入 think 标签
        self.in_think_tag = True
        self.think_buffer = parts[1]
        
        return outputs
    
    def _handle_think_end(self, token: str) -> List[Dict[str, Any]]:
        """处理 think 标签结束"""
        outputs = []
        parts = token.split("</think>", 1)
        
        # 完成 think 标签
        self.think_buffer += parts[0]
        
        # 输出思考内容
        if self.think_buffer.strip():
            outputs.append({
                "type": "thought",
                "content": filter_think_content(self.think_buffer.strip())
            })
        
        # 重置状态
        self.in_think_tag = False
        self.think_buffer = ""
        
        # </think> 标签之后的内容作为普通文本
        after = parts[1]
        if after.strip():
            self.full_text += after
            outputs.append({"type": "token", "content": after, "full_text": self.full_text})
        
        return outputs
    
    def reset(self):
        """重置处理器状态"""
        self.in_think_tag = False
        self.think_buffer = ""
        self.full_text = ""
