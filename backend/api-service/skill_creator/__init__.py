"""
技能初始化模块
自动从 skill_creator 目录发现并注册所有技能
"""

import os
import importlib
from pathlib import Path
from logger import get_logger
from .registry import get_skill_registry

logger = get_logger(__name__)

# 全局标记：技能是否已初始化
_skills_initialized = False


def discover_skills():
    """
    自动发现 skill_creator 目录下的所有技能
    
    规则：
    1. 遍历 skill_creator 下的所有子目录
    2. 查找包含 SKILL.md 的目录
    3. 查找对应的 Python 实现文件
    4. 自动注册技能
    """
    registry = get_skill_registry()
    skill_creator_dir = Path(__file__).parent
    
    # 技能映射：目录名 -> 模块名 -> 函数名
    skill_mapping = {
        "resume_polishing": ("resume_polishing", "polish_resume"),
        "jd_keyword_match": ("jd_keyword_match", "match_jd_keywords"),
        "resume_score": ("resume_score", "score_resume"),
    }
    
    for skill_dir_name, (module_name, func_name) in skill_mapping.items():
        skill_dir = skill_creator_dir / skill_dir_name
        
        if not skill_dir.exists():
            logger.warning(f"Skill directory not found: {skill_dir}")
            continue
        
        # 检查是否有 SKILL.md
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            logger.warning(f"SKILL.md not found in: {skill_dir}")
            continue
        
        # 检查是否有 Python 实现
        skill_py = skill_dir / f"{module_name}.py"
        if not skill_py.exists():
            logger.warning(f"Skill implementation not found: {skill_py}")
            continue
        
        try:
            # 动态导入模块
            module_path = f"skill_creator.{skill_dir_name}.{module_name}"
            module = importlib.import_module(module_path)
            
            # 获取函数
            skill_func = getattr(module, func_name)
            
            # 注册技能
            registry.register(skill_dir_name, skill_func)
            logger.info(f"Registered skill: {skill_dir_name}")
            
        except Exception as e:
            logger.error(f"Failed to register skill {skill_dir_name}: {e}")
    
    logger.info(f"All skills initialized. Total: {len(registry.list_skills())}")
    return registry


def init_skills():
    """初始化技能系统（带防重复初始化）"""
    global _skills_initialized
    
    if _skills_initialized:
        logger.debug("Skills already initialized, skipping...")
        return get_skill_registry()
    
    registry = discover_skills()
    _skills_initialized = True
    return registry
