from __future__ import annotations
from typing import Dict, Any, Callable, Optional
from logger import get_logger

logger = get_logger(__name__)


_skill_router_agent = None


def _get_skill_router():
    """懒加载 SkillRouterAgent"""
    global _skill_router_agent
    if _skill_router_agent is None:
        from .skill_router_agent import SkillRouterAgent
        _skill_router_agent = SkillRouterAgent()
    return _skill_router_agent


class SkillRegistry:
    """技能注册表"""
    
    _instance: Optional[SkillRegistry] = None
    
    def __new__(cls) -> SkillRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.skills = {}
        return cls._instance
    
    def register(self, name: str, skill_func: Callable) -> None:
        """注册技能"""
        self.skills[name] = skill_func
        logger.info(f"Skill registered: {name}")
    
    def unregister(self, name: str) -> None:
        """注销技能"""
        if name in self.skills:
            del self.skills[name]
            logger.info(f"Skill unregistered: {name}")
    
    def get(self, name: str) -> Optional[Callable]:
        """获取技能"""
        return self.skills.get(name)
    
    def list_skills(self) -> Dict[str, Callable]:
        """列出所有已注册的技能"""
        return self.skills.copy()
    
    def execute(self, name: str, **kwargs) -> Any:
        """执行技能"""
        if name not in self.skills:
            raise ValueError(f"Skill not found: {name}")
        
        skill_func = self.skills[name]
        logger.info(f"Executing skill: {name} with args: {list(kwargs.keys())}")
        
        try:
            result = skill_func(**kwargs)
            logger.info(f"Skill {name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Skill {name} execution failed: {e}", exc_info=True)
            raise


class SkillExecutor:
    """技能执行器"""
    
    def __init__(self, registry: Optional[SkillRegistry] = None):
        self.registry = registry or SkillRegistry()
    
    def execute(self, skill_name: str, **kwargs) -> Any:
        """执行指定技能"""
        return self.registry.execute(skill_name, **kwargs)
    
    def execute_with_context(self, skill_name: str, context: Dict[str, Any]) -> Any:
        """带上下文执行技能"""
        return self.registry.execute(skill_name, **context)
    
    def list_available_skills(self) -> list:
        """列出可用技能"""
        return list(self.registry.list_skills().keys())
    
    def auto_select_skill(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据用户输入自动选择技能（使用三层路由 Agent）
        
        Args:
            user_input: 用户输入
            context: 上下文信息（如是否有上传文件等）
        
        Returns:
            路由结果字典，包含 skill, confidence, source, reason
        """
        router = _get_skill_router()
        result = router.route(user_input, context)
        
        logger.info(
            f"Auto-selected skill: {result.skill} (confidence={result.confidence:.2f}, source={result.source})"
        )
        
        return {
            "skill": result.skill,
            "confidence": result.confidence,
            "source": result.source,
            "reason": result.reason,
            "params": result.params
        }
    
    def execute_auto(self, user_input: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        自动选择并执行技能
        
        Args:
            user_input: 用户输入
            context: 上下文信息
            **kwargs: 技能参数
        
        Returns:
            包含路由信息和执行结果的字典
        """
        # 自动选择技能
        route_result = self.auto_select_skill(user_input, context)
        skill_name = route_result["skill"]
        
        # 执行技能
        execution_result = self.execute(skill_name, **kwargs)
        
        return {
            "route": route_result,
            "result": execution_result
        }


def get_skill_registry() -> SkillRegistry:
    """获取技能注册表单例"""
    return SkillRegistry()


def get_skill_executor() -> SkillExecutor:
    """获取技能执行器"""
    return SkillExecutor()
