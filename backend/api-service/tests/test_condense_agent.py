"""精简简历 skill 与 planner 条件步骤测试

覆盖：
- should_condense 阈值与否定指令
- ResumeCondenseAgent.run 成功/失败回落/结果清理（mock LLM）
- ResumeCondenseExecutor 执行器契约
- MasterAgent 模板计划：超长简历插入 resume_condense，润色依赖正确
"""
from unittest.mock import MagicMock

import pytest

from agents.skills.resume_agents.condense_agent import (
    ResumeCondenseAgent,
    should_condense,
    CONDENSE_THRESHOLD,
)
from agents.mas.executors.resume_condense import ResumeCondenseExecutor
from agents.mas.planner import MasterAgent
from agents.mas.state import IntentResult, INTENT_RESUME_OPTIMIZE


# ---------- should_condense ----------

def test_should_condense_threshold():
    assert not should_condense("简" * CONDENSE_THRESHOLD)
    assert should_condense("简" * (CONDENSE_THRESHOLD + 1))


def test_should_condense_empty():
    assert not should_condense("")
    assert not should_condense("   ")


@pytest.mark.parametrize("phrase", [
    "不要精简内容",
    "不用精简",
    "无需删减",
    "请保留全部内容",
    "全文保留",
    "do not condense my resume",
    "keep all content",
])
def test_should_condense_negation(phrase):
    long_text = phrase + "，" + "内容" * 1500
    assert not should_condense(long_text)


def test_should_condense_triggers_on_plain_long_text():
    assert should_condense("工作经历 " + "负责系统设计与优化 " * 400)


# ---------- ResumeCondenseAgent.run ----------

def _state(resume, jd=None):
    return {"resume": resume, "jd": jd, "current_step": "started"}


def test_condense_run_success():
    mock_llm = MagicMock()
    # 注意：内容不得以"这是/以下是"等解释性前缀开头，否则会被 _clean_result 清理
    mock_llm.generate.return_value = "工作经历：负责核心系统架构设计与性能优化" * 8
    agent = ResumeCondenseAgent(llm=mock_llm)

    state = _state("原" * 3000)
    result = agent.run(state)

    assert result["condensed_resume"]
    assert result["original_length"] == 3000
    assert result["condensed_length"] < 3000
    assert mock_llm.generate.called
    # max_new_tokens 显式传参（防默认截断）
    assert mock_llm.generate.call_args.kwargs.get("max_new_tokens") == 2048


def test_condense_run_empty_output_falls_back_to_original():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = ""
    agent = ResumeCondenseAgent(llm=mock_llm)

    state = _state("原" * 3000)
    result = agent.run(state)

    # 失败回落原文，不阻断流程
    assert result["condensed_resume"] == "原" * 3000
    assert result.get("skipped") is True


def test_condense_run_too_short_output_falls_back():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "太短"
    agent = ResumeCondenseAgent(llm=mock_llm)
    result = agent.run(_state("原" * 3000))
    assert result.get("skipped") is True


def test_clean_result_strips_artifacts():
    raw = (
        "好的，以下是精简结果：\n"
        "```\n工作经历：保留\n```"
    )
    cleaned = ResumeCondenseAgent._clean_result(raw)
    assert "好的" not in cleaned
    assert "```" not in cleaned
    assert "工作经历：保留" in cleaned


def test_clean_result_strips_think_tags():
    raw = "<think>让我分析一下\n这是思考过程</think>精简后的正文内容" * 2
    cleaned = ResumeCondenseAgent._clean_result(raw)
    assert "<think>" not in cleaned


# ---------- Executor ----------

def test_executor_name_and_not_streamable():
    assert ResumeCondenseExecutor.name == "resume_condense"
    assert ResumeCondenseExecutor.streamable is False
    assert "resume" in ResumeCondenseExecutor.required_inputs


def test_executor_run_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "精简结果正文" * 20
    executor = ResumeCondenseExecutor(llm=mock_llm)

    result = executor.run({
        "step_id": "step_3",
        "resume": "原" * 3000,
        "jd": "",
    })
    assert result.success
    assert result.data["condensed_resume"]
    assert result.data["original_length"] == 3000
    assert result.data["condensed_length"] < 3000


def test_executor_missing_resume_fails():
    executor = ResumeCondenseExecutor(llm=MagicMock())
    result = executor.run({"step_id": "s"})
    assert not result.success


# ---------- 超长简历 map-reduce 分段 ----------

def test_long_resume_condensed_segment_by_segment():
    """7000 字简历（>3800 单段上限）分 2 段 map，每段独立调用 LLM"""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = lambda prompt, **kw: (
        "MAP_A 精简结果：保留核心项目与量化成果，主导订单交易系统重构，"
        "服务可用性提升至四个九，内容足够长以通过有效性校验。" * 3
        if "第 1/2 部分" in prompt
        else "MAP_B 精简结果：保留教育经历与技能栈，精通 Python 与 Go，"
        "具备云原生与团队管理经验，内容足够长以通过有效性校验。" * 3
    )
    agent = ResumeCondenseAgent(llm=mock_llm)
    result = agent.run({"resume": "长" * 7000})

    assert mock_llm.generate.call_count == 2
    assert "MAP_A" in result["condensed_resume"]
    assert "MAP_B" in result["condensed_resume"]
    assert result["original_length"] == 7000
    # 简历全文不再被 resume[:6000] 截断：两段 prompt 合计覆盖全部 7000 字
    prompts = [c.args[0] for c in mock_llm.generate.call_args_list]
    assert any("第 2/2 部分" in p for p in prompts)


def test_segment_failure_falls_back_to_original():
    """任一段精简失败（返回空）→ 整体回退原文，不阻断"""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = lambda prompt, **kw: (
        "" if "第 2/2 部分" in prompt
        else "第一段有效精简结果：保留核心项目与量化成果，主导订单交易系统重构，"
        "服务可用性提升至四个九，内容足够长以通过校验。" * 3
    )
    agent = ResumeCondenseAgent(llm=mock_llm)
    result = agent.run({"resume": "长" * 7000})

    assert result.get("skipped") is True
    assert result["condensed_resume"] == "长" * 7000


def test_reduce_triggered_when_merged_map_too_long():
    """两段 map 合并后仍 >5000 字：触发 reduce 再压缩一轮"""
    mapped_piece = "压" * 2600  # 每段 2600，两段合并 5202 > 5000
    reduce_piece = "REDUCE 终稿：核心内容精简版，保留关键项目与量化成果，长度达标通过校验。" * 4
    mock_llm = MagicMock()
    # 2 次 map + reduce 阶段合并稿被切成 2 块 → 2 次 reduce
    mock_llm.generate.side_effect = [mapped_piece, mapped_piece, reduce_piece, reduce_piece]
    agent = ResumeCondenseAgent(llm=mock_llm)

    result = agent.run({"resume": "长" * 7000})
    assert mock_llm.generate.call_count == 4
    assert "REDUCE" in result["condensed_resume"]


def test_llm_exception_falls_back_to_original():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("llm unavailable")
    agent = ResumeCondenseAgent(llm=mock_llm)
    result = agent.run({"resume": "长" * 3000})
    assert result.get("skipped") is True
    assert result["condensed_resume"] == "长" * 3000


def test_executor_treats_fallback_as_success():
    """跨层契约：精简回退原文必须写入 state，executor 视为成功而非失败
    （否则反思器会误判并触发无谓重试）"""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("llm unavailable")
    executor = ResumeCondenseExecutor(llm=mock_llm)
    result = executor.run({"step_id": "s1", "resume": "长" * 3000})

    assert result.success is True
    assert result.data["condensed_resume"] == "长" * 3000


# ---------- Planner 条件插入 ----------

def _plan(resume, user_input=""):
    master = MasterAgent(llm=None)
    intent = IntentResult(
        name=INTENT_RESUME_OPTIMIZE, confidence=1.0,
        source="preset", reason="test", is_complex=True,
    )
    return master.plan(user_input, intent, {"resume": resume})


def test_plan_long_resume_inserts_condense_step():
    plan = _plan("简" * 3000)
    agents = [s.agent for s in plan.steps]
    assert agents == ["resume_score", "jd_match", "resume_condense", "resume_polish"]

    polish = next(s for s in plan.steps if s.agent == "resume_polish")
    assert "step_3" in polish.depends_on
    # 润色依赖全部前置步骤
    assert set(polish.depends_on) == {"step_1", "step_2", "step_3"}


def test_plan_short_resume_skips_condense():
    plan = _plan("张三，3年Java经验")
    agents = [s.agent for s in plan.steps]
    assert agents == ["resume_score", "jd_match", "resume_polish"]
    polish = next(s for s in plan.steps if s.agent == "resume_polish")
    assert set(polish.depends_on) == {"step_1", "step_2"}


def test_plan_long_resume_with_negation_skips_condense():
    plan = _plan("不要精简，保留全部内容：" + "详" * 3000)
    agents = [s.agent for s in plan.steps]
    assert "resume_condense" not in agents


def test_plan_reads_resume_from_context_when_user_input_empty():
    # resume 路由 user_input 为空串、resume 在 context 中
    master = MasterAgent(llm=None)
    intent = IntentResult(
        name=INTENT_RESUME_OPTIMIZE, confidence=1.0,
        source="preset", reason="test", is_complex=True,
    )
    plan = master.plan("", intent, {"resume": "长" * 3000})
    assert any(s.agent == "resume_condense" for s in plan.steps)


def test_planner_agent_whitelist_includes_condense():
    master = MasterAgent(llm=None, executor_catalog="")
    assert master._is_valid_agent("resume_condense")
    assert not master._is_valid_agent("not_an_agent")
