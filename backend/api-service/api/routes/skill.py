"""
技能执行相关路由
/skill/*

技能引擎，支持按名称执行指定技能或自动识别用户意图。
"""

from __future__ import annotations
from fastapi import APIRouter

from logger import get_logger
from services.agent_service import AgentService
from .common import SkillExecuteRequest

router = APIRouter(tags=["技能"])
logger = get_logger(__name__)

agent_service = AgentService()


@router.post(
    "/execute",
    summary="执行指定技能",
    description="""
按名称执行指定技能。

**常用技能名称：**
- `resume_score` - 简历评分
- `resume_suggest` - 简历优化建议
- `jd_parse` - JD 解析
- `match` - 简历-JD 匹配度分析
    """,
    responses={
        200: {"description": "执行成功"},
        400: {"description": "技能不存在或参数错误"},
        500: {"description": "执行错误"},
    },
)
async def execute_skill(request: SkillExecuteRequest):
    try:
        result = agent_service.execute_skill(
            skill_name=request.skill_name,
            **request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Skill execute error: {e}")
        return {"error": str(e)}


@router.post(
    "/execute/auto",
    summary="自动识别并执行技能",
    description="根据用户输入自动判断意图，选择合适的技能执行。返回结果包含路由决策信息和执行结果。",
    responses={
        200: {
            "description": "执行成功",
            "example": {
                "route": {
                    "skill": "resume-polishing",
                    "confidence": 0.95,
                    "source": "keyword",
                    "reason": "关键词匹配: 润色, 简历"
                },
                "result": "技能执行结果..."
            }
        },
    }
)
async def execute_skill_auto(user_input: str):
    try:
        result = agent_service.execute_skill_auto(user_input)
        return result
    except Exception as e:
        logger.error(f"Skill auto execute error: {e}")
        return {"error": str(e)}


@router.post(
    "/route",
    summary="意图路由（仅识别，不执行）",
    description="根据用户输入进行意图识别，返回路由决策信息，但不执行技能。用于前端展示或调试。",
    responses={
        200: {
            "description": "路由成功",
            "example": {
                "skill": "resume-polishing",
                "confidence": 0.95,
                "source": "keyword",
                "reason": "关键词匹配: 润色, 简历",
                "params": {}
            }
        },
    }
)
async def route_intent(user_input: str):
    try:
        route_result = agent_service.skill_executor.auto_select_skill(user_input)
        return route_result
    except Exception as e:
        logger.error(f"Intent route error: {e}")
        return {"error": str(e)}


@router.get(
    "/list",
    summary="列出所有可用技能",
    description="返回当前系统注册的所有可用技能及其描述",
)
async def list_skills():
    try:
        skills = agent_service.list_skills()
        return {"skills": skills}
    except Exception as e:
        logger.error(f"List skills error: {e}")
        return {"error": str(e)}
