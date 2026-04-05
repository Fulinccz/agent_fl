"""
Agent Registry - Agent 工厂和注册中心

职责：
- 统一创建 Provider 实例
- 管理可用的模型提供者
- 支持工具注册和发现
- 为 LangChain 集成预留接口

设计模式：
- 工厂模式：统一创建对象
- 单例模式：全局唯一的 Registry
- 注册表模式：动态管理组件

使用方式：
    # 基础用法（向后兼容）
    agent = get_agent(provider="local", model="...")
    
    # 新用法（推荐）
    from agents.registry import AgentRegistry
    registry = AgentRegistry()
    provider = registry.get_provider("local")
    
    # 获取工具
    parser_tool = registry.get_tool("resume_parser", provider=provider)
"""

from __future__ import annotations

from logger import get_logger
from typing import Optional, Dict, Any
from pathlib import Path

# 导入新的架构组件
from .providers.local import LocalProvider, LocalAgent
from .providers.online import OnlineProvider, OpenAIAgent
from .core.base_tool import ToolRegistry as BaseToolRegistry

logger = get_logger(__name__)


class AgentRegistry:
    """
    Agent 注册中心（增强版）
    
    功能：
    - Provider 工厂：创建和管理模型提供者
    - Tool 发现：查找和实例化工具
    - 配置管理：集中管理配置信息
    
    未来扩展：
    - 支持 LangChain Agent 创建
    - 支持多 Provider 负载均衡
    - 支持 Plugin 系统
    """
    
    _instance = None
    _providers: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._default_model = None
        self._tool_registry = BaseToolRegistry()
        logger.info("AgentRegistry 初始化完成")
    
    @staticmethod
    def _get_default_local_model() -> str:
        """获取默认的本地模型路径"""
        root = Path(__file__).resolve().parents[3]
        serving_dir = root / "ai" / "models" / "model_serving"
        
        if not serving_dir.exists() or not serving_dir.is_dir():
            raise FileNotFoundError(f"默认本地模型目录未找到：{serving_dir}")
        
        for item in serving_dir.iterdir():
            if item.is_dir():
                return str(item)
        
        raise FileNotFoundError(
            f"ai/models/model_serving 下未找到任何模型文件夹，请添加模型后重试。"
        )
    
    def get_provider(
        self, 
        provider: str = "local", 
        model: Optional[str] = None,
        **kwargs
    ):
        """
        获取模型提供者实例
        
        Args:
            provider: 提供者类型 ('local', 'online', 'openai', 'cloud')
            model: 模型名称或路径
            **kwargs: 额外参数（如 api_key）
            
        Returns:
            Provider 实例 (LocalProvider 或 OnlineProvider)
        """
        import time
        call_time = time.strftime('%H:%M:%S')
        normalized = (provider or "").strip().lower()
        
        logger.info(f"[{call_time}] === get_provider 被调用 ===")
        logger.info(f"[{call_time}] provider={normalized}, model={model}")
        
        if normalized in {"online", "openai", "cloud"}:
            logger.info(f"[{call_time}] Using Online provider (model={model or 'gpt-3.5-turbo'})")
            
            api_key = kwargs.get('api_key')
            return OnlineProvider(api_key=api_key, model=model or "gpt-3.5-turbo")
        
        chosen_model = model or self._get_default_local_model()
        logger.info(f"[{call_time}] Using Local provider (model={chosen_model})")
        
        return LocalProvider(model_name=chosen_model)
    
    def get_tool(self, tool_name: str, provider=None):
        """
        获取工具实例
        
        Args:
            tool_name: 工具名称
            provider: 模型提供者（可选）
            
        Returns:
            Tool 实例
        """
        return self._tool_registry.get_tool(tool_name, provider=provider)
    
    def list_tools(self) -> list:
        """列出所有可用工具"""
        return self._tool_registry.list_tools()
    
    def create_resume_agent(self, provider=None):
        """
        创建简历优化 Agent（未来 LangChain 版本）
        
        TODO: 实现 LangChain ReAct Agent
              - 自动选择合适的工具
              - 多步骤工作流编排
              - 记忆和上下文管理
        """
        tools = [
            self.get_tool("resume_parser", provider),
            self.get_tool("resume_scorer", provider),
            self.get_tool("resume_optimizer", provider),
            self.get_tool("text_polisher", provider),
        ]
        
        # 返回工具列表（未来将包装为 LangChain Agent）
        return {
            "provider": provider,
            "tools": [t.name for t in tools if t],
            "description": "简历优化 Agent（包含解析、打分、优化、润色能力）"
        }


# 向后兼容：保留原有的 get_agent 函数
def get_agent(provider: str = "local", model: Optional[str] = None):
    """
    获取 Agent 实例（向后兼容接口）
    
    此函数保持与旧代码完全兼容，内部调用新的 AgentRegistry
    
    Args:
        provider: 提供者类型
        model: 模型名称
        
    Returns:
        Provider 实例
    """
    registry = AgentRegistry()
    return registry.get_provider(provider=provider, model=model)


# 便捷导出
__all__ = [
    'AgentRegistry',
    'get_agent',
    'LocalProvider',
    'LocalAgent',
    'OnlineProvider',
    'OpenAIAgent',
]
