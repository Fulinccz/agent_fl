"""页面入口档案

前端三个页面（简历分析 / 简历生成 / 面试攻略）共用一个优化端点，
但每个入口需要的意图、任务计划、可用执行器与提示词指令都不同。
本模块是这份映射的唯一事实来源，路由、planner、执行器都从这里取，
避免"分页逻辑"散落在三处各写一套。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.mas.state import (
    INTENT_INTERVIEW_PREP,
    INTENT_RESUME_ANALYZE,
    INTENT_RESUME_OPTIMIZE,
)


@dataclass(frozen=True)
class PageStep:
    """页面计划中的步骤模板"""
    agent: str
    task: str
    # 仅当用户提供了 JD 时才执行（简历生成 / 面试攻略不依赖 JD）
    requires_jd: bool = False


@dataclass(frozen=True)
class PageProfile:
    """单个页面的完整行为描述"""
    key: str
    intent: str
    steps: tuple[PageStep, ...]
    # 该页面允许的执行器白名单（LLM 规划的步骤也按此过滤）
    agents: frozenset[str]
    # 注入到各 agent 提示词的场景指令
    directive: str


PAGE_PROFILES: Dict[str, PageProfile] = {
    "analyze": PageProfile(
        key="analyze",
        intent=INTENT_RESUME_ANALYZE,
        steps=(
            PageStep("resume_score", "对简历进行多维度评分并定位问题"),
            PageStep("jd_match", "对照目标岗位 JD 分析差距并给出改进建议", requires_jd=True),
        ),
        agents=frozenset({"resume_score", "jd_match"}),
        directive=(
            "当前入口是【简历分析】：以招聘顾问的视角做诊断，只指出问题与差距，"
            "禁止改写或重写简历；每条评估与建议都必须引用简历原文的具体表述。"
        ),
    ),
    "generate": PageProfile(
        key="generate",
        intent=INTENT_RESUME_OPTIMIZE,
        steps=(
            PageStep("resume_polish", "基于简历内容生成可直接投递的成品简历"),
        ),
        agents=frozenset({"resume_condense", "resume_polish"}),
        directive=(
            "当前入口是【简历生成】：输出可直接投递的成品简历正文，"
            "不要输出点评、建议、说明或 Markdown 标题装饰；"
            "保持经历与事实不变，用主导性动词和量化成果强化表述。"
        ),
    ),
    "interview": PageProfile(
        key="interview",
        intent=INTENT_INTERVIEW_PREP,
        steps=(
            PageStep("interview_prep", "分析简历与目标岗位，生成分维度面试题库与答题要点"),
        ),
        agents=frozenset({"interview_prep"}),
        directive=(
            "当前入口是【面试攻略】：以目标岗位面试官的身份出题，"
            "题目必须扣住简历中的具体经历与 JD 要求，答题要点给出可复述的结构化提示。"
        ),
    ),
}


def get_page_profile(page: Optional[str]) -> Optional[PageProfile]:
    """按页面 key 取档案；未知/空值返回 None（走旧的全流程优化）"""
    return PAGE_PROFILES.get((page or "").strip().lower())


def apply_page_context(context: Dict[str, Any], page: Optional[str]) -> Optional[PageProfile]:
    """把页面档案写入黑板上下文（page 供 planner 取计划，directive 供执行器注入提示词）"""
    profile = get_page_profile(page)
    if profile is not None:
        context["page"] = profile.key
        context["directive"] = profile.directive
    return profile


def page_agents(page: Optional[str]) -> Optional[frozenset[str]]:
    """页面允许的执行器白名单；无页面档案时返回 None（不做限制）"""
    profile = get_page_profile(page)
    return profile.agents if profile else None
