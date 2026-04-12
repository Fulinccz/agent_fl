"""
技能注册表和执行引擎
集中管理所有简历优化技能
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Optional
from logger import get_logger

logger = get_logger(__name__)


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
    
    def auto_select_skill(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        根据用户输入自动选择技能
        
        Args:
            user_input: 用户输入
            context: 上下文信息（如是否有上传文件等）
        
        Returns:
            选中的技能名称
        """
        # 技能关键词映射
        skill_keywords = {
            "resume-polishing": [
                "润色", "优化", "改进", "修改", "调整",
                "大白话", "口语化", "专业", "话术",
                "polish", "improve", "optimize"
            ],
            "jd-keyword-match": [
                "匹配", "JD", "岗位", "招聘", "关键词",
                "ATS", "筛选", "匹配度",
                "match", "job description", "keyword"
            ],
            "resume-score": [
                "评分", "打分", "评估", "评价", "分数",
                "多少分", "怎么样", "如何",
                "score", "rate", "evaluate", "evaluation"
            ],
        }
        
        user_input_lower = user_input.lower()
        
        # 统计每个技能的匹配度
        skill_scores = {}
        for skill_name, keywords in skill_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                skill_scores[skill_name] = score
        
        # 如果有匹配的技能，返回得分最高的
        if skill_scores:
            best_skill = max(skill_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Auto-selected skill: {best_skill} (score: {skill_scores[best_skill]})")
            return best_skill
        
        # 默认返回简历润色技能
        logger.info("No matching skill found, using default: resume-polishing")
        return "resume-polishing"
    
    def execute_auto(self, user_input: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        自动选择并执行技能
        
        Args:
            user_input: 用户输入
            context: 上下文信息
            **kwargs: 技能参数
        
        Returns:
            技能执行结果
        """
        # 自动选择技能
        skill_name = self.auto_select_skill(user_input, context)
        
        # 执行技能
        return self.execute(skill_name, **kwargs)


def get_skill_registry() -> SkillRegistry:
    """获取技能注册表单例"""
    return SkillRegistry()


def get_skill_executor() -> SkillExecutor:
    """获取技能执行器"""
    return SkillExecutor()
