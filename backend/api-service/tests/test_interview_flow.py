"""面试攻略链路：意图关键词 / 规划模板 / SSE 事件契约 / 预置意图映射"""
from agents.mas.master import MasterOrchestrator
from agents.mas.planner import MasterAgent
from agents.mas.state import (
    IntentResult,
    StepResult,
    TaskPlan,
    TaskStep,
    INTENT_INTERVIEW_PREP,
    INTENT_RESUME_OPTIMIZE,
)
from api.routes.common import build_preset_intent
from skillhub.keyword_matcher import KeywordMatcher


def test_keyword_match_interview_skill():
    result = KeywordMatcher().match("帮我准备面试")

    assert result.skill == "interview-prep"
    assert result.confidence > 0.3


def test_exact_pattern_interview_guide():
    result = KeywordMatcher().match("面试攻略")

    assert result.skill == "interview-prep"
    assert result.confidence == 1.0


def test_planner_accepts_interview_agent_without_catalog():
    master = MasterAgent(executor_catalog="")

    assert master._is_valid_agent("interview_prep") is True


def test_planner_template_is_single_step():
    plan = MasterAgent().plan(
        "面试攻略", IntentResult(name=INTENT_INTERVIEW_PREP, confidence=1.0)
    )

    assert plan.intent == INTENT_INTERVIEW_PREP
    assert len(plan.steps) == 1
    assert plan.steps[0].agent == "interview_prep"


def test_legacy_events_emit_interview_event():
    orch = MasterOrchestrator.__new__(MasterOrchestrator)  # 跳过初始化，避免加载模型/向量
    guide = {"position": "后端工程师", "dimensions": []}
    result = StepResult(
        step_id="step_1", agent="interview_prep", success=True,
        data={"interview_guide": guide},
    )
    step = TaskStep(step_id="step_1", agent="interview_prep", task="生成面试题库")
    plan = TaskPlan(goal="面试攻略", intent=INTENT_INTERVIEW_PREP)

    events = list(orch._legacy_events(step, result, plan))

    assert events == [{"type": "interview", "data": {"interview_guide": guide}}]


def test_build_preset_intent_mapping():
    interview = build_preset_intent("interview-prep")
    assert interview.name == INTENT_INTERVIEW_PREP
    assert interview.is_complex is False
    assert interview.source == "preset"

    default = build_preset_intent(None)
    assert default.name == INTENT_RESUME_OPTIMIZE
    assert default.is_complex is True

    unknown = build_preset_intent("不存在的意图")
    assert unknown.name == INTENT_RESUME_OPTIMIZE
