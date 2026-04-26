"""
技能执行相关路由
/skill/*
"""

from __future__ import annotations
from fastapi import APIRouter

from logger import get_logger
from services.agent_service import AgentService
from .common import SkillExecuteRequest

router = APIRouter()
logger = get_logger(__name__)

agent_service = AgentService()


@router.post("/execute")
async def execute_skill(request: SkillExecuteRequest):
    """执行指定技能"""
    try:
        result = agent_service.execute_skill(
            skill_name=request.skill_name,
            **request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Skill execute error: {e}")
        return {"error": str(e)}


@router.post("/execute/auto")
async def execute_skill_auto(user_input: str):
    """自动识别并执行技能"""
    try:
        result = agent_service.execute_skill_auto(user_input)
        return result
    except Exception as e:
        logger.error(f"Skill auto execute error: {e}")
        return {"error": str(e)}


@router.get("/list")
async def list_skills():
    """列出所有可用技能"""
    try:
        skills = agent_service.list_skills()
        return {"skills": skills}
    except Exception as e:
        logger.error(f"List skills error: {e}")
        return {"error": str(e)}
