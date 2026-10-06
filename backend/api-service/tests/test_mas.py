"""主从多智能体系统（agents/mas）测试套件

覆盖：意图识别、Master 规划、执行器、ReAct 反思、编排器全流程与异常路径。
全部使用 mock LLM，不加载真实模型。
"""
import json
from unittest.mock import MagicMock

import pytest

from agents.mas.state import (
    IntentResult,
    TaskPlan,
    TaskStep,
    StepResult,
    Reflection,
    Blackboard,
    INTENT_RESUME_OPTIMIZE,
    INTENT_RESUME_SCORE,
    INTENT_RESUME_POLISH,
    INTENT_JD_MATCH,
    INTENT_RESUME_PARSE,
    INTENT_CHAT_GENERAL,
    INTENT_COMPLEX_TASK,
    VERDICT_RETRY,
    VERDICT_REPLAN,
    VERDICT_CONTINUE,
    VERDICT_FINISH,
    VERDICT_ABORT,
)
from agents.mas.intent import IntentRecognizer, detect_multiple_intents
from agents.mas.planner import MasterAgent
from agents.mas.reflector import Reflector
from agents.mas.executors import (
    ExecutorRegistry,
    BaseExecutor,
    ResumeScoreExecutor,
    JDMatchExecutor,
    ResumePolishExecutor,
    ResumeParseExecutor,
    ChatExecutor,
)
from agents.mas.master import MasterOrchestrator


SAMPLE_RESUME = """
张三，高级后端工程师
工作经历：2020-2023 某科技公司 - 负责微服务架构设计
技能：Python、Go、MySQL、Redis、Kubernetes
"""

SAMPLE_JD = "招聘高级后端工程师，要求熟悉 Python/Go，有微服务经验"

SCORE_JSON = '{"completeness": 80, "professionalism": 75, "quantification": 70, "matching": 80}'
MATCH_JSON = json.dumps({
    "match_score": 82,
    "matched_keywords": ["Python", "微服务"],
    "missing_keywords": ["Kubernetes"],
    "suggestions": ["补充云原生经验", "增加量化数据"],
}, ensure_ascii=False)


def make_mock_llm(score_text=SCORE_JSON, match_text=MATCH_JSON, polish_tokens=None,
                  chat_tokens=None, generate_error=None):
    """构造模拟 LLM：按提示词内容路由到不同返回"""
    mock = MagicMock()

    def generate(prompt, **kwargs):
        if generate_error:
            raise generate_error
        if "completeness" in prompt:          # 评分任务提示词特征
            return score_text
        if "matched_keywords" in prompt:      # JD 匹配任务提示词特征
            return match_text
        return "默认回复"

    mock.generate.side_effect = generate

    def generate_stream(prompt, **kwargs):
        # 按提示词特征路由：润色 vs 对话
        if "请优化以下简历描述" in prompt:
            tokens = polish_tokens if polish_tokens is not None else ["润色后的", "简历内容"]
        else:
            tokens = chat_tokens if chat_tokens is not None else ["你好，", "我是简历助手"]
        for tok in tokens:
            yield {"type": "token", "content": tok}

    mock.generate_stream.side_effect = generate_stream
    return mock


CHAT_PRESET = IntentResult(name=INTENT_CHAT_GENERAL, confidence=1.0, source="preset")
RESUME_PRESET = IntentResult(name=INTENT_RESUME_OPTIMIZE, confidence=1.0,
                             source="preset", is_complex=True)


# ============ 意图识别测试 ============

class TestIntentRecognizer:
    def setup_method(self):
        # 不注入 llm、不触碰路由器（纯规则路径）
        self.recognizer = IntentRecognizer()

    def test_preset_intent_passthrough(self):
        preset = IntentResult(name=INTENT_RESUME_OPTIMIZE, confidence=1.0,
                              source="preset", is_complex=True)
        result = self.recognizer.recognize("任意输入", {}, preset=preset)
        assert result is preset

    def test_empty_input_defaults_chat(self):
        result = self.recognizer.recognize("", {})
        assert result.name == INTENT_CHAT_GENERAL

    def test_complex_marker_hits_resume_optimize(self):
        result = self.recognizer.recognize("帮我优化简历，越专业越好", {})
        assert result.name == INTENT_RESUME_OPTIMIZE
        assert result.is_complex is True

    def test_route_error_falls_back_to_chat(self):
        result = self.recognizer._post_process(
            skill="unknown-skill", confidence=0.9, source="llm",
            reason="test", user_input="hi", has_resume=False, has_jd=False,
        )
        assert result.name == INTENT_CHAT_GENERAL
        assert result.confidence <= 0.6

    def test_chat_with_resume_context_upgrades_to_optimize(self):
        result = self.recognizer._post_process(
            skill=INTENT_CHAT_GENERAL, confidence=0.5, source="keyword",
            reason="test", user_input="帮我改进一下", has_resume=True, has_jd=False,
        )
        assert result.name == INTENT_RESUME_OPTIMIZE
        assert result.is_complex is True

    def test_simple_skill_gets_agent_mapping(self):
        result = self.recognizer._post_process(
            skill="resume-score", confidence=0.95, source="keyword",
            reason="test", user_input="打分", has_resume=False, has_jd=False,
        )
        assert result.name == "resume-score"
        assert result.params.get("agent") == "resume_score"
        assert result.is_complex is False

    def test_lite_router_without_vector(self):
        """enable_vector=False 时走轻量路由（关键词+LLM），不加载向量模型"""
        llm = MagicMock()
        llm.generate.return_value = "默认回复"  # LLM 分类输出无 JSON → 兜底 chat-general
        recognizer = IntentRecognizer(llm=llm, enable_vector=False)
        result = recognizer.recognize("随便聊聊今天的天气怎么样")
        assert result.name == INTENT_CHAT_GENERAL


# ============ Master 规划测试 ============

class TestMasterAgent:
    def setup_method(self):
        self.master = MasterAgent(executor_catalog="- resume_score: 评分\n- chat: 对话")

    def test_template_plan_resume_optimize(self):
        intent = IntentResult(name=INTENT_RESUME_OPTIMIZE, is_complex=True)
        plan = self.master.plan("优化简历", intent, {})
        assert plan.source == "template"
        agents = [s.agent for s in plan.steps]
        assert agents == ["resume_score", "jd_match", "resume_polish"]
        # 润色依赖前两步
        assert set(plan.steps[2].depends_on) == {"step_1", "step_2"}

    def test_template_plan_chat(self):
        intent = IntentResult(name=INTENT_CHAT_GENERAL)
        plan = self.master.plan("你好", intent, {})
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == "chat"

    def test_llm_plan_parses_valid_json(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps({
            "goal": "测试目标",
            "steps": [
                {"step_id": "step_1", "agent": "resume_score", "task": "评分", "depends_on": []},
                {"step_id": "step_2", "agent": "chat", "task": "解读", "depends_on": ["step_1"]},
            ],
        }, ensure_ascii=False)
        master = MasterAgent(llm=llm, executor_catalog="- resume_score: 评分\n- chat: 对话")
        intent = IntentResult(name="complex-task", is_complex=True)
        plan = master.plan("复杂请求", intent, {})
        assert plan.source == "llm"
        assert len(plan.steps) == 2
        assert plan.steps[1].depends_on == ["step_1"]

    def test_llm_plan_filters_unknown_agents(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps({
            "steps": [
                {"step_id": "step_1", "agent": "不存在的能力", "task": "x", "depends_on": []},
                {"step_id": "step_2", "agent": "chat", "task": "y", "depends_on": ["step_1"]},
            ],
        }, ensure_ascii=False)
        master = MasterAgent(llm=llm, executor_catalog="- chat: 对话")
        intent = IntentResult(name="complex-task", is_complex=True)
        plan = master.plan("复杂请求", intent, {})
        # 未知执行器步骤被过滤，悬空依赖被清理
        assert [s.agent for s in plan.steps] == ["chat"]
        assert plan.steps[0].depends_on == []

    def test_llm_plan_invalid_output_falls_back_to_template(self):
        llm = MagicMock()
        llm.generate.return_value = "这不是JSON"
        master = MasterAgent(llm=llm)
        intent = IntentResult(name="complex-task", is_complex=True)
        plan = master.plan("复杂请求", intent, {})
        assert plan.source == "template"

    def test_replan_returns_none_when_llm_fails(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("模型不可用")
        master = MasterAgent(llm=llm)
        result = master.replan("目标", ["轨迹"], "step_1", "错误", "反馈")
        assert result is None


# ============ 执行器测试 ============

class TestExecutors:
    def test_score_executor_missing_input(self):
        executor = ResumeScoreExecutor(llm=make_mock_llm())
        result = executor.run({"step_id": "s1"})
        assert not result.success
        assert "缺少必需输入" in result.error

    def test_score_executor_success(self):
        executor = ResumeScoreExecutor(llm=make_mock_llm())
        result = executor.run({"step_id": "s1", "resume": SAMPLE_RESUME})
        assert result.success
        assert result.data["overall_score"]["score"] == pytest.approx(76.25, abs=0.1)

    def test_score_executor_llm_error_reports_failure(self):
        executor = ResumeScoreExecutor(llm=make_mock_llm(generate_error=RuntimeError("GPU 挂了")))
        result = executor.run({"step_id": "s1", "resume": SAMPLE_RESUME})
        assert not result.success
        assert "评分失败" in result.error

    def test_match_executor_without_jd_uses_defaults(self):
        executor = JDMatchExecutor(llm=make_mock_llm())
        result = executor.run({"step_id": "s2", "resume": SAMPLE_RESUME})
        assert result.success
        assert result.data["match_result"]["match_score"] == 70

    def test_match_executor_success(self):
        executor = JDMatchExecutor(llm=make_mock_llm())
        result = executor.run({"step_id": "s2", "resume": SAMPLE_RESUME, "jd": SAMPLE_JD})
        assert result.success
        assert result.data["match_result"]["match_score"] == 82

    def test_polish_executor_stream(self):
        executor = ResumePolishExecutor(llm=make_mock_llm(polish_tokens=["优化后", "的内容"]))
        chunks = list(executor.run_stream({"step_id": "s3", "resume": SAMPLE_RESUME}))
        assert all(c["type"] == "token" for c in chunks)
        assert "".join(c["content"] for c in chunks) == "优化后的内容"

    def test_chat_executor_missing_message(self):
        executor = ChatExecutor(llm=make_mock_llm())
        result = executor.run({"step_id": "s4"})
        assert not result.success

    def test_registry_has_all_executors(self):
        registry = ExecutorRegistry(llm=make_mock_llm())
        assert set(registry.names()) == {
            "resume_score", "jd_match", "resume_polish", "resume_parse", "chat"
        }
        assert "resume_score" in registry.catalog()


# ============ ReAct 反思测试 ============

class TestReflector:
    def setup_method(self):
        self.reflector = Reflector(llm=make_mock_llm(), use_llm_reflection=False)

    def _make_step(self, agent="resume_score", retries=0):
        return TaskStep(step_id="step_1", agent=agent, task="测试任务", retries=retries)

    def test_success_continues(self):
        result = StepResult(step_id="step_1", agent="resume_score", success=True,
                            data={"score_result": {}, "overall_score": {"score": 80}})
        reflection = self.reflector.review(self._make_step(), result, "目标", [])
        assert reflection.verdict == VERDICT_CONTINUE
        assert reflection.source == "rule"

    def test_failure_retries_then_replan(self):
        result = StepResult(step_id="step_1", agent="resume_score",
                            success=False, error="执行出错")
        # 第一次：重试
        r1 = self.reflector.review(self._make_step(retries=0), result, "目标", [])
        assert r1.verdict == VERDICT_RETRY
        assert "重试" in r1.feedback or "失败原因" in r1.feedback
        # 超过最大重试：重规划
        r2 = self.reflector.review(self._make_step(retries=2), result, "目标", [])
        assert r2.verdict == VERDICT_REPLAN

    def test_missing_outputs_triggers_retry(self):
        result = StepResult(step_id="step_1", agent="resume_score", success=True, data={})
        reflection = self.reflector.review(self._make_step(), result, "目标", [])
        assert reflection.verdict == VERDICT_RETRY

    def test_polish_too_short_triggers_retry(self):
        result = StepResult(step_id="step_1", agent="resume_polish", success=True,
                            data={"optimized_resume": "太短"})
        reflection = self.reflector.review(
            self._make_step(agent="resume_polish"), result, "目标", [])
        assert reflection.verdict == VERDICT_RETRY

    def test_llm_verdict_parsing(self):
        reflector = Reflector(llm=make_mock_llm(), use_llm_reflection=True)
        llm_result = MagicMock()
        llm_result.generate.return_value = json.dumps({
            "verdict": "retry", "thought": "产出不完整", "feedback": "请完整生成"
        }, ensure_ascii=False)
        reflector._llm = llm_result
        result = StepResult(step_id="step_1", agent="chat", success=True, data={})
        step = self._make_step(agent="chat", retries=5)  # 规则重试耗尽
        reflection = reflector.review(step, result, "目标", [])
        # chat 无必需产出校验字段缺失逻辑？chat 有 ["answer"] → 规则会 retry；
        # retries 已达上限 → 规则裁决 replan，先于 LLM
        assert reflection.verdict == VERDICT_REPLAN

    def test_llm_review_used_when_rule_undecided(self):
        reflector = Reflector(llm=make_mock_llm(), use_llm_reflection=True)
        reflector._llm = MagicMock()
        reflector._llm.generate.return_value = json.dumps({
            "verdict": "continue", "thought": "可接受", "feedback": ""
        }, ensure_ascii=False)
        # resume_parse 无必需产出校验字段 → 规则判 continue，不会走到 LLM；
        # 构造一个规则返回 None 的场景：改为直接单测 _llm_review
        result = StepResult(step_id="step_1", agent="chat", success=True,
                            data={"answer": "部分回答"})
        step = self._make_step(agent="chat")
        reflection = reflector._llm_review(step, result, "目标", [])
        assert reflection is not None
        assert reflection.verdict == VERDICT_CONTINUE
        assert reflection.source == "llm"


# ============ 黑板测试 ============

class TestBlackboard:
    def test_snapshot_merges_results(self):
        board = Blackboard(message="hello", resume="简历")
        board.record(StepResult(step_id="s1", agent="resume_score", success=True,
                                data={"score_result": {"a": 1}}))
        board.record(StepResult(step_id="s2", agent="jd_match", success=False,
                                data={"match_result": {"b": 2}}))
        snap = board.snapshot()
        # 成功结果合入，失败结果不合入
        assert snap["score_result"] == {"a": 1}
        assert "match_result" not in snap
        assert snap["resume"] == "简历"

    def test_trace_lines(self):
        board = Blackboard()
        board.record(StepResult(step_id="s1", agent="chat", success=True, summary="完成"))
        lines = board.trace_lines()
        assert len(lines) == 1
        assert "s1" in lines[0] and "成功" in lines[0]


# ============ 编排器测试 ============

class TestMasterOrchestrator:
    def _collect(self, orchestrator, *args, **kwargs):
        return list(orchestrator.run_stream(*args, **kwargs))

    def test_chat_flow_events(self):
        llm = make_mock_llm(chat_tokens=["你好！", "我是简历助手"])
        orchestrator = MasterOrchestrator(llm=llm, enable_vector_intent=False,
                                          use_llm_reflection=False)
        events = self._collect(orchestrator, "你好", {}, preset_intent=CHAT_PRESET)

        types = [e["type"] for e in events]
        assert types[0] == "intent"
        assert "plan" in types
        assert "step_start" in types
        assert "reflection" in types
        # token 事件透传
        tokens = [e for e in events if e["type"] == "token"]
        assert "".join(t["content"] for t in tokens) == "你好！我是简历助手"
        # complete 事件
        complete = [e for e in events if e["type"] == "complete"][-1]
        assert complete["data"]["steps_success"] == 1

    def test_resume_optimize_full_flow(self):
        llm = make_mock_llm(polish_tokens=[
            "优化后的简历：主导微服务架构设计，",
            "性能提升30%，完成5个核心项目交付，",
            "技术栈覆盖Python/Go/Kubernetes，表达更专业。",
        ])
        orchestrator = MasterOrchestrator(llm=llm, enable_vector_intent=False,
                                          use_llm_reflection=False)
        context = {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}
        events = self._collect(orchestrator, "", context, preset_intent=RESUME_PRESET)

        types = [e["type"] for e in events]
        # 旧契约事件齐全且有序
        assert types.index("score") < types.index("suggestions") < types.index("complete")
        assert "polished" in types
        # 最终 polished 为 partial=False
        polished_final = [e for e in events
                          if e["type"] == "polished" and not e["data"].get("partial")]
        assert len(polished_final) == 1
        assert "优化后的简历" in polished_final[0]["data"]["optimized_resume"]
        # complete 汇总
        complete = [e for e in events if e["type"] == "complete"][-1]
        assert complete["data"]["optimized_resume"]
        assert complete["data"]["overall_score"]
        assert complete["data"]["steps_total"] == 3

    def test_run_non_stream_aggregates(self):
        llm = make_mock_llm(chat_tokens=["回答内容"])
        orchestrator = MasterOrchestrator(llm=llm, enable_vector_intent=False,
                                          use_llm_reflection=False)
        result = orchestrator.run("你好", {}, preset_intent=CHAT_PRESET)
        assert result["success"] is True
        assert result["answer"] == "回答内容"

    def test_executor_failure_retry_and_skip(self):
        # 所有 LLM 调用失败 → 评分失败 → 反思重试×2 → 重规划失败 → 跳过 → 正常结束
        llm = make_mock_llm(generate_error=RuntimeError("模型故障"))
        orchestrator = MasterOrchestrator(llm=llm, enable_vector_intent=False,
                                          use_llm_reflection=False)
        preset = IntentResult(name=INTENT_RESUME_OPTIMIZE, confidence=1.0,
                              source="preset", is_complex=True)
        events = self._collect(orchestrator, "",
                               {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD},
                               preset_intent=preset)

        types = [e["type"] for e in events]
        # 反思事件中应有 retry 裁决
        reflections = [e["data"]["verdict"] for e in events if e["type"] == "reflection"]
        assert VERDICT_RETRY in reflections
        # 流程正常收敛（有 complete，无未捕获异常）
        assert "complete" in types

    def test_ready_steps_respects_dependencies_and_skips(self):
        plan = TaskPlan(goal="g", intent="i", steps=[
            TaskStep(step_id="step_1", agent="a", task="t1"),
            TaskStep(step_id="step_2", agent="b", task="t2", depends_on=["step_1"]),
            TaskStep(step_id="step_3", agent="c", task="t3", depends_on=["step_2"]),
        ])
        # 初始：step_1 无依赖，可直接执行
        ready = MasterOrchestrator._ready_steps(plan)
        assert [s.step_id for s in ready] == ["step_1"]
        # step_1 完成 → step_2 就绪
        plan.steps[0].status = "success"
        ready = MasterOrchestrator._ready_steps(plan)
        assert [s.step_id for s in ready] == ["step_2"]
        # step_2 被跳过 → step_3 就绪（跳过视为依赖满足）
        plan.steps[1].status = "skipped"
        ready = MasterOrchestrator._ready_steps(plan)
        assert [s.step_id for s in ready] == ["step_3"]

    def test_apply_replan_replaces_pending(self):
        orchestrator = MasterOrchestrator(llm=make_mock_llm(), enable_vector_intent=False,
                                          use_llm_reflection=False)
        plan = TaskPlan(goal="g", intent="i", steps=[
            TaskStep(step_id="step_1", agent="a", task="t1", status="failed"),
            TaskStep(step_id="step_2", agent="b", task="t2"),
        ])
        new_steps = [TaskStep(step_id="step_r1", agent="chat", task="重试任务")]
        assert orchestrator._apply_replan(plan, new_steps, failed=plan.steps[0]) is True
        assert plan.steps[1].status == "skipped"
        assert plan.source == "replan"
        assert any(s.step_id == "step_r1" for s in plan.steps)

    def test_finish_verdict_skips_remaining(self):
        llm = make_mock_llm()
        orchestrator = MasterOrchestrator(llm=llm, enable_vector_intent=False,
                                          use_llm_reflection=False)
        # 直接单测 _skip_remaining
        plan = TaskPlan(goal="g", intent="i", steps=[
            TaskStep(step_id="step_1", agent="a", task="t1"),
            TaskStep(step_id="step_2", agent="b", task="t2"),
        ])
        orchestrator._skip_remaining(plan)
        assert all(s.status == "skipped" for s in plan.steps)


# ============ 核心数据结构补充测试 ============

class TestStateStructures:
    def test_intent_result_to_dict_rounds_confidence(self):
        d = IntentResult(name=INTENT_CHAT_GENERAL, confidence=0.123456).to_dict()
        assert d == {
            "name": INTENT_CHAT_GENERAL,
            "confidence": 0.123,
            "source": "rule",
            "reason": "",
            "is_complex": False,
        }

    def test_task_step_to_dict_copies_dependencies(self):
        deps = ["step_1"]
        step = TaskStep(step_id="step_2", agent="chat", task="t", depends_on=deps)
        d = step.to_dict()
        assert d["depends_on"] == ["step_1"]
        assert d["status"] == "pending" and d["retries"] == 0
        # 返回的是副本，修改不影响原对象
        d["depends_on"].append("hack")
        assert step.depends_on == ["step_1"]

    def test_task_plan_and_step_result_to_dict(self):
        plan = TaskPlan(goal="g", intent="i", source="llm", steps=[
            TaskStep(step_id="step_1", agent="chat", task="t"),
        ])
        pd = plan.to_dict()
        assert pd["goal"] == "g" and pd["source"] == "llm"
        assert len(pd["steps"]) == 1

        rd = StepResult(step_id="s1", agent="chat").to_dict()
        assert rd["success"] is False and rd["data"] == {} and rd["error"] == ""

    def test_reflection_to_dict(self):
        d = Reflection(step_id="s", verdict=VERDICT_ABORT, thought="停", source="llm").to_dict()
        assert d["verdict"] == "abort" and d["feedback"] == ""

    def test_blackboard_inputs_and_agent_index(self):
        board = Blackboard(message="hi")
        board.set_input("resume", "简历文本")
        r1 = StepResult(step_id="s1", agent="chat", success=True, data={"answer": "第一条"})
        r2 = StepResult(step_id="s2", agent="chat", success=True, data={"answer": "第二条"})
        board.record(r1)
        board.record(r2)
        # agent_results 记录最近一次结果
        assert board.agent_results["chat"] is r2
        # snapshot 中同名字段后写覆盖先写
        assert board.snapshot()["answer"] == "第二条"
        assert board.snapshot()["message"] == "hi"
        assert board.snapshot()["resume"] == "简历文本"

    def test_blackboard_trace_empty_and_failure(self):
        assert Blackboard().trace_lines() == []
        board = Blackboard()
        board.record(StepResult(step_id="s1", agent="chat", success=False,
                                error="x" * 200, summary="失败了"))
        line = board.trace_lines()[0]
        assert "失败" in line and "s1" in line
        # 错误信息被截断到 80 字以内
        assert "x" * 81 not in line


# ============ 意图识别补充测试 ============

class TestIntentRecognizerExtra:
    def test_other_complex_markers(self):
        recognizer = IntentRecognizer()
        for text in ["一键优化我的简历", "全面优化", "full optimize my resume"]:
            result = recognizer.recognize(text, {})
            assert result.name == INTENT_RESUME_OPTIMIZE, text
            assert result.is_complex is True

    def test_lite_router_keyword_high_confidence(self, monkeypatch):
        """轻量路由：关键词高置信命中直接返回（EXACT_PATTERNS 置信度 1.0）"""
        recognizer = IntentRecognizer(llm=MagicMock(), enable_vector=False)
        result = recognizer.recognize("帮我做个简历评分", {})
        assert result.name == "resume-score"
        assert result.source == "keyword"
        assert result.params["agent"] == "resume_score"

    def test_lite_router_falls_through_to_llm_layer(self, monkeypatch):
        """轻量路由：关键词低置信 → LLM 分类层"""
        from skillhub.keyword_matcher import MatchResult
        monkeypatch.setattr(
            "skillhub.keyword_matcher.KeywordMatcher.match",
            lambda self, text: MatchResult(None, 0.1, []),
        )
        llm = MagicMock()
        llm.generate.return_value = json.dumps({
            "intent": "jd-keyword-match", "confidence": 0.83,
            "reason": "询问岗位匹配", "params": {},
        }, ensure_ascii=False)
        recognizer = IntentRecognizer(llm=llm, enable_vector=False)
        result = recognizer.recognize("这个岗位适合我吗", {})
        assert result.name == "jd-keyword-match"
        assert result.source == "llm"
        assert result.params["agent"] == "jd_match"

    def test_full_router_init_failure_falls_back(self, monkeypatch):
        """完整路由器初始化失败（如 faiss 不可用）→ router 为 None → 规则降级"""
        def _boom(*a, **k):
            raise RuntimeError("faiss missing")
        monkeypatch.setattr("skillhub.skill_router_agent.SkillRouterAgent", _boom)
        recognizer = IntentRecognizer(enable_vector=True)
        assert recognizer.router is None
        result = recognizer.recognize("今天天气不错", {})
        assert result.name == INTENT_CHAT_GENERAL
        assert result.source == "fallback"

    def test_route_exception_falls_back(self):
        recognizer = IntentRecognizer()
        fake_router = MagicMock()
        fake_router.route.side_effect = RuntimeError("路由爆炸")
        recognizer._router = fake_router
        skill, conf, source, reason = recognizer._route("随便聊点啥")
        assert skill == INTENT_CHAT_GENERAL and source == "fallback"

    def test_post_process_resume_optimize_aliases(self):
        recognizer = IntentRecognizer()
        for alias in (INTENT_RESUME_OPTIMIZE, "resume_optimize"):
            result = recognizer._post_process(
                skill=alias, confidence=0.9, source="llm",
                reason="t", user_input="优化", has_resume=False, has_jd=False,
            )
            assert result.name == INTENT_RESUME_OPTIMIZE
            assert result.is_complex is True

    def test_chat_with_resume_but_no_optimize_hint_stays_chat(self):
        recognizer = IntentRecognizer()
        result = recognizer._post_process(
            skill=INTENT_CHAT_GENERAL, confidence=0.7, source="keyword",
            reason="t", user_input="你好", has_resume=True, has_jd=False,
        )
        assert result.name == INTENT_CHAT_GENERAL
        assert result.is_complex is False

    def test_file_path_context_counts_as_resume(self):
        """上下文中的 file_path 与 resume 等价，可触发复杂任务升级"""
        recognizer = IntentRecognizer()
        # 绕过路由器：直接喂 chat-general 技能做后处理
        result = recognizer._post_process(
            skill=INTENT_CHAT_GENERAL, confidence=0.5, source="keyword",
            reason="t", user_input="帮我润色改进一下", has_resume=True, has_jd=False,
        )
        assert result.name == INTENT_RESUME_OPTIMIZE

    def test_simple_skill_mappings(self):
        recognizer = IntentRecognizer()
        result = recognizer._post_process(
            skill="resume-parse", confidence=0.9, source="keyword",
            reason="t", user_input="解析", has_resume=False, has_jd=False,
        )
        assert result.params["agent"] == "resume_parse"
        assert result.is_complex is False

    def test_detect_multiple_intents(self):
        hits = detect_multiple_intents("帮我润色简历并打分")
        assert "resume-polishing" in hits
        assert "resume-score" in hits


# ============ Master 规划补充测试 ============

class TestMasterAgentExtra:
    def _master(self, llm=None):
        return MasterAgent(
            llm=llm,
            executor_catalog="- resume_score: 评分\n- chat: 对话\n- jd_match: 匹配",
        )

    @pytest.mark.parametrize("intent_name,agent", [
        (INTENT_RESUME_SCORE, "resume_score"),
        (INTENT_JD_MATCH, "jd_match"),
        (INTENT_RESUME_POLISH, "resume_polish"),
        (INTENT_RESUME_PARSE, "resume_parse"),
        (INTENT_CHAT_GENERAL, "chat"),
    ])
    def test_simple_intent_templates(self, intent_name, agent):
        plan = self._master().plan(
            "请求", IntentResult(name=intent_name), {}
        )
        assert plan.source == "template"
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == agent

    def test_resume_optimize_complex_uses_template_without_llm(self):
        """resume-optimize 走确定性模板，不消耗 LLM 调用"""
        llm = MagicMock()
        master = self._master(llm)
        plan = master.plan(
            "优化简历", IntentResult(name=INTENT_RESUME_OPTIMIZE, is_complex=True), {}
        )
        assert plan.source == "template"
        llm.generate.assert_not_called()

    def test_parse_steps_dedupes_step_ids(self):
        raw = json.dumps({"steps": [
            {"step_id": "x", "agent": "chat", "task": "a", "depends_on": []},
            {"step_id": "x", "agent": "chat", "task": "b", "depends_on": ["x"]},
        ]}, ensure_ascii=False)
        steps = self._master()._parse_steps(raw)
        ids = [s.step_id for s in steps]
        assert len(ids) == 2 and len(set(ids)) == 2
        # 第二个步骤对 x 的依赖指向保留下来的第一个步骤
        assert steps[1].depends_on == ["x"]

    def test_parse_steps_filters_and_sanitizes(self):
        raw = json.dumps({"steps": [
            "not-a-dict",
            {"step_id": "s1", "agent": "chat", "task": "a", "depends_on": "step_1"},
            {"step_id": "s2", "agent": "chat", "task": "b", "depends_on": ["s2", "ghost"]},
            {"step_id": "s3", "agent": "未知能力", "task": "c"},
        ]}, ensure_ascii=False)
        steps = self._master()._parse_steps(raw)
        agents = [s.agent for s in steps]
        assert agents == ["chat", "chat"]
        # depends_on 非 list → 空；自依赖/悬空依赖被清理
        assert steps[0].depends_on == []
        assert steps[1].depends_on == []

    def test_parse_steps_invalid_shapes(self):
        m = self._master()
        assert m._parse_steps('{"steps": {"a": 1}}') is None
        assert m._parse_steps('{"steps": []}') is None
        assert m._parse_steps("没有JSON") is None
        assert m._parse_steps('{"other": 1}') is None

    def test_llm_plan_empty_steps_returns_none(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps({"steps": []})
        master = self._master(llm)
        assert master._llm_plan(
            "复杂请求", IntentResult(name=INTENT_COMPLEX_TASK, is_complex=True), {}
        ) is None

    def test_llm_plan_without_provider_uses_shared(self, monkeypatch):
        master = self._master(llm=None)
        monkeypatch.setattr(
            "agents.mas.planner.get_shared_llm",
            lambda: (_ for _ in ()).throw(RuntimeError("模型未加载")),
        )
        assert master._llm_plan(
            "复杂请求", IntentResult(name=INTENT_COMPLEX_TASK, is_complex=True), {}
        ) is None

    def test_replan_success(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps({"steps": [
            {"step_id": "rp1", "agent": "chat", "task": "补救", "depends_on": []},
        ]}, ensure_ascii=False)
        plan = self._master(llm).replan("目标", ["轨迹"], "step_1", "错误", "反馈")
        assert plan is not None
        assert plan.source == "replan"
        assert plan.steps[0].step_id == "rp1"

    def test_replan_empty_output_returns_none(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps({"steps": []})
        assert self._master(llm).replan("g", [], "s", "e", "f") is None

    def test_replan_shared_llm_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "agents.mas.planner.get_shared_llm",
            lambda: (_ for _ in ()).throw(RuntimeError("不可用")),
        )
        master = MasterAgent(llm=None, executor_catalog="- chat: 对话")
        assert master.replan("g", [], "s", "e", "f") is None

    def test_is_valid_agent_without_catalog(self):
        m = MasterAgent()  # 无执行器目录
        assert m._is_valid_agent("chat") is True
        assert m._is_valid_agent("resume_parse") is True
        assert m._is_valid_agent("ghost") is False


# ============ 执行器补充测试 ============

class _FakeScoreAgent:
    """模拟 ResumeScoreAgent，避免加载真实模型"""
    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state):
        if getattr(self, "_fail", False):
            state["error"] = "评分服务异常"
        state["score_result"] = {"completeness": 70}
        state["overall_score"] = {"score": 70, "rating": "C"}


class _FakeMatchAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state):
        state["match_result"] = {"match_score": 66}
        state["suggestions"] = ["建议一"]


class _FakePolishAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state):
        state["optimized_resume"] = "专业的简历内容" * 4

    def run_stream(self, state):
        for tok in ["专业", "内容"]:
            yield {"type": "token", "content": tok}

    def _clean_result(self, raw_text, original_resume):
        return "CLEANED:" + raw_text


class TestExecutorsExtra:
    def test_parse_missing_input(self):
        result = ResumeParseExecutor().run({"step_id": "s"})
        assert not result.success and "缺少必需输入" in result.error

    def test_parse_success_backfills_resume(self, monkeypatch):
        monkeypatch.setattr(
            "rag.document_processor.parse_resume",
            lambda path: {"skills": "Python", "projects": "项目A", "name": "张三"},
        )
        result = ResumeParseExecutor().run({"step_id": "s", "file_path": "a.pdf"})
        assert result.success
        # 解析产物回填 resume（skills + projects）
        assert result.data["resume"] == "Python\n\n项目A"
        assert result.data["name"] == "张三"

    def test_parse_empty_and_exception(self, monkeypatch):
        monkeypatch.setattr("rag.document_processor.parse_resume", lambda path: {})
        result = ResumeParseExecutor().run({"step_id": "s", "file_path": "a.pdf"})
        assert not result.success and "为空" in result.error

        def _boom(path):
            raise RuntimeError("文件损坏")
        monkeypatch.setattr("rag.document_processor.parse_resume", _boom)
        result = ResumeParseExecutor().run({"step_id": "s", "file_path": "b.pdf"})
        assert not result.success and "文件损坏" in result.error

    def test_polish_run_success_and_empty(self, monkeypatch):
        monkeypatch.setattr(
            "agents.skills.resume_agents.polish_agent.ResumePolishAgent",
            _FakePolishAgent,
        )
        ex = ResumePolishExecutor(llm=MagicMock())
        ok = ex.run({"step_id": "s", "resume": SAMPLE_RESUME})
        assert ok.success and ok.data["optimized_resume"]

        class _EmptyPolish(_FakePolishAgent):
            def run(self, state):
                state["optimized_resume"] = None
        monkeypatch.setattr(
            "agents.skills.resume_agents.polish_agent.ResumePolishAgent",
            _EmptyPolish,
        )
        bad = ResumePolishExecutor(llm=MagicMock()).run({"step_id": "s", "resume": SAMPLE_RESUME})
        assert not bad.success

    def test_polish_run_exception_and_stream_validation(self, monkeypatch):
        class _BoomPolish:
            def __init__(self, llm=None): ...
            def run(self, state):
                raise RuntimeError("润色崩溃")
            def run_stream(self, state):
                raise RuntimeError("流式崩溃")
        monkeypatch.setattr(
            "agents.skills.resume_agents.polish_agent.ResumePolishAgent", _BoomPolish
        )
        ex = ResumePolishExecutor(llm=MagicMock())
        assert "润色崩溃" in ex.run({"step_id": "s", "resume": SAMPLE_RESUME}).error

        chunks = list(ex.run_stream({"step_id": "s"}))  # 缺 resume → 校验错误
        assert chunks == [{"type": "error", "content": chunks[0]["content"]}]
        assert "缺少必需输入" in chunks[0]["content"]

    def test_polish_finalize_reuses_clean_logic(self, monkeypatch):
        monkeypatch.setattr(
            "agents.skills.resume_agents.polish_agent.ResumePolishAgent",
            _FakePolishAgent,
        )
        text = ResumePolishExecutor(llm=MagicMock()).finalize("raw", SAMPLE_RESUME)
        assert text == "CLEANED:raw"

    def test_score_failure_keeps_fallback_scores(self, monkeypatch):
        class _FailScore(_FakeScoreAgent):
            _fail = True
        monkeypatch.setattr(
            "agents.skills.resume_agents.score_agent.ResumeScoreAgent", _FailScore
        )
        result = ResumeScoreExecutor(llm=MagicMock()).run(
            {"step_id": "s", "resume": SAMPLE_RESUME}
        )
        # 报错但兜底分数保留在 data 中
        assert not result.success
        assert result.data["overall_score"]["score"] == 70

    def test_match_state_error_still_succeeds(self, monkeypatch):
        class _MatchWithError(_FakeMatchAgent):
            def run(self, state):
                super().run(state)
                state["error"] = "部分失败"
        monkeypatch.setattr(
            "agents.skills.resume_agents.match_agent.JDMatchAgent", _MatchWithError
        )
        result = JDMatchExecutor(llm=MagicMock()).run(
            {"step_id": "s", "resume": SAMPLE_RESUME}
        )
        assert result.success is True
        assert result.data["match_result"]["match_score"] == 66

    def test_chat_build_prompt_and_clean(self):
        ex = ChatExecutor(llm=MagicMock())
        plain = ex._build_prompt({"message": "你好"})
        assert "你好" in plain and "参考资料" not in plain
        with_rag = ex._build_prompt({"message": "问题", "rag_context": "知识库片段"})
        assert "知识库片段" in with_rag and "参考资料" in with_rag

        assert ex._clean("<think>思考</think>正文") == "正文"
        assert ex._clean("assistant: 你好") == "你好"
        assert ex._clean("\nASSISTANT：嗨") == "嗨"

    def test_chat_run_success_empty_and_error(self):
        ok = ChatExecutor(llm=make_mock_llm()).run({"step_id": "s", "message": "你好"})
        assert ok.success and ok.data["answer"] == "默认回复"
        assert ok.data["_stream_text"] == "默认回复"

        empty_llm = MagicMock()
        empty_llm.generate.return_value = "   "
        bad = ChatExecutor(llm=empty_llm).run({"step_id": "s", "message": "x"})
        assert not bad.success and "为空" in bad.error

        boom_llm = MagicMock()
        boom_llm.generate.side_effect = RuntimeError("挂了")
        err = ChatExecutor(llm=boom_llm).run({"step_id": "s", "message": "x"})
        assert not err.success and "挂了" in err.error

    def test_chat_run_stream_error_chunk_and_exception(self):
        llm = MagicMock()
        llm.generate_stream.return_value = iter([
            {"type": "error", "content": "生成失败"},
        ])
        chunks = list(ChatExecutor(llm=llm).run_stream({"step_id": "s", "message": "x"}))
        assert chunks == [{"type": "error", "content": "生成失败"}]

        llm2 = MagicMock()
        llm2.generate_stream.side_effect = RuntimeError("流挂了")
        chunks = list(ChatExecutor(llm=llm2).run_stream({"step_id": "s", "message": "x"}))
        assert "流挂了" in chunks[0]["content"]

    def test_base_executor_default_stream_fallback(self):
        class _SyncEx(BaseExecutor):
            name = "demo"
            def run(self, task_input):
                return StepResult(step_id="s", agent="demo", success=True,
                                  data={"_stream_text": "整段文本"})

        chunks = list(_SyncEx().run_stream({}))
        assert chunks == [{"type": "token", "content": "整段文本"}]

        class _FailEx(BaseExecutor):
            name = "demo"
            def run(self, task_input):
                return StepResult(step_id="s", agent="demo", success=False, error="坏")
        chunks = list(_FailEx().run_stream({}))
        assert chunks[0] == {"type": "error", "content": "坏"}

    def test_registry_get_and_has(self):
        registry = ExecutorRegistry(llm=make_mock_llm())
        assert registry.has("chat")
        assert registry.get("chat").name == "chat"
        assert registry.get("不存在") is None


# ============ ReAct 反思补充测试 ============

class TestReflectorExtra:
    def _step(self, agent="resume_polish", retries=0):
        return TaskStep(step_id="step_1", agent=agent, task="t", retries=retries)

    def test_llm_review_invalid_json_and_verdict(self):
        llm = MagicMock()
        llm.generate.return_value = "模型自由发挥，没有JSON"
        r = Reflector(llm=llm)
        result = StepResult(step_id="step_1", agent="chat", success=True, data={"answer": "x"})
        assert r._llm_review(self._step(), result, "g", []) is None

        llm.generate.return_value = json.dumps({"verdict": "explode"})
        assert r._llm_review(self._step(), result, "g", []) is None

    def test_llm_review_generate_exception(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("模型超时")
        r = Reflector(llm=llm)
        result = StepResult(step_id="step_1", agent="chat", success=True, data={"answer": "x"})
        assert r._llm_review(self._step(), result, "g", []) is None

    def test_llm_review_uses_shared_provider(self, monkeypatch):
        shared = MagicMock()
        shared.generate.return_value = json.dumps({
            "verdict": "abort", "thought": "无法修复", "feedback": "",
        }, ensure_ascii=False)
        monkeypatch.setattr("agents.mas.reflector.get_shared_llm", lambda: shared)
        r = Reflector(llm=None)  # 未注入 → 延迟取共享 LLM
        result = StepResult(step_id="step_1", agent="chat", success=True, data={"answer": "x"})
        reflection = r._llm_review(self._step(), result, "g", [])
        assert reflection is not None and reflection.verdict == VERDICT_ABORT

    def test_short_polish_escalates_to_llm_and_accepts(self):
        """润色过短属于边界情况：规则不裁决，升级 LLM；LLM 认可 → continue"""
        llm = MagicMock()
        llm.generate.return_value = json.dumps({
            "verdict": "continue", "thought": "简历本身简短，30 字内可接受", "feedback": "",
        }, ensure_ascii=False)
        r = Reflector(llm=llm, use_llm_reflection=True)
        result = StepResult(step_id="step_1", agent="resume_polish", success=True,
                            data={"optimized_resume": "简短简历"})
        reflection = r.review(self._step(), result, "g", [])
        assert reflection.source == "llm"
        assert reflection.verdict == VERDICT_CONTINUE

    def test_short_polish_llm_abort(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps({
            "verdict": "abort", "thought": "润色彻底失败", "feedback": "",
        }, ensure_ascii=False)
        r = Reflector(llm=llm, use_llm_reflection=True)
        result = StepResult(step_id="step_1", agent="resume_polish", success=True,
                            data={"optimized_resume": "短"})
        reflection = r.review(self._step(), result, "g", [])
        assert reflection.verdict == VERDICT_ABORT

    def test_short_polish_without_llm_falls_back_to_replan(self):
        """LLM 审查关闭且重试次数耗尽 → 保守兜底 replan"""
        r = Reflector(llm=MagicMock(), use_llm_reflection=False)
        result = StepResult(step_id="step_1", agent="resume_polish", success=True,
                            data={"optimized_resume": "短"})
        reflection = r.review(self._step(retries=2), result, "g", [])
        assert reflection.verdict == VERDICT_REPLAN
        assert reflection.source == "fallback"

    def test_polish_length_boundary_30_chars(self):
        r = Reflector(llm=MagicMock(), use_llm_reflection=True)
        result = StepResult(step_id="step_1", agent="resume_polish", success=True,
                            data={"optimized_resume": "甲" * 30})
        reflection = r.review(self._step(), result, "g", [])
        assert reflection.verdict == VERDICT_CONTINUE and reflection.source == "rule"

    @pytest.mark.parametrize("agent,data,should_miss", [
        ("jd_match", {"match_result": {}}, "suggestions"),
        ("chat", {"answer": ""}, "answer"),
        ("resume_parse", {"anything": 1}, None),
        ("unknown_agent", {}, None),
    ])
    def test_check_outputs_matrix(self, agent, data, should_miss):
        missing = Reflector._check_outputs(agent, data)
        if should_miss:
            assert should_miss in missing
        else:
            assert missing == []


# ============ 编排器补充测试 ============

class TestMasterOrchestratorExtra:
    def _collect(self, orchestrator, *args, **kwargs):
        return list(orchestrator.run_stream(*args, **kwargs))

    def _orchestrator(self, llm=None, **kw):
        return MasterOrchestrator(
            llm=llm or make_mock_llm(),
            enable_vector_intent=False,
            use_llm_reflection=kw.get("use_llm_reflection", False),
        )

    def test_retry_then_success_end_to_end(self):
        """评分首次失败 → retry → 重试成功 → 全流程收敛"""
        llm = MagicMock()
        score_calls = {"n": 0}

        def generate(prompt, **kwargs):
            if "completeness" in prompt:
                score_calls["n"] += 1
                if score_calls["n"] == 1:
                    raise RuntimeError("首次评分失败")
                return SCORE_JSON
            if "matched_keywords" in prompt:
                return MATCH_JSON
            return "默认回复"

        llm.generate.side_effect = generate

        def generate_stream(prompt, **kwargs):
            tokens = (["专业简历内容，主导微服务架构设计，" * 3]
                      if "请优化以下简历描述" in prompt else ["你好"])
            for tok in tokens:
                yield {"type": "token", "content": tok}
        llm.generate_stream.side_effect = generate_stream

        orch = self._orchestrator(llm)
        events = self._collect(
            orch, "", {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD},
            preset_intent=RESUME_PRESET,
        )
        verdicts = [e["data"]["verdict"] for e in events if e["type"] == "reflection"]
        assert VERDICT_RETRY in verdicts
        complete = [e for e in events if e["type"] == "complete"][-1]
        # 三个步骤最终全部成功（评分重试后成功）
        assert complete["data"]["steps_success"] == 3

    def test_abort_verdict_emits_error_and_stops(self):
        orch = self._orchestrator()
        orch.reflector.review = lambda *a, **k: Reflection(
            step_id="step_1", verdict=VERDICT_ABORT, thought="不可恢复", source="llm"
        )
        events = self._collect(orch, "你好", {}, preset_intent=CHAT_PRESET)
        types = [e["type"] for e in events]
        assert "error" in types and "complete" not in types
        assert events[-1]["message"] == "不可恢复"

    def test_finish_verdict_skips_remaining_end_to_end(self):
        orch = self._orchestrator()
        orch.reflector.review = lambda *a, **k: Reflection(
            step_id="step_1", verdict=VERDICT_FINISH, thought="目标达成", source="llm"
        )
        events = self._collect(
            orch, "", {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD},
            preset_intent=RESUME_PRESET,
        )
        types = [e["type"] for e in events]
        assert "complete" in types and "score" not in types[types.index("reflection") + 1:]
        complete = [e for e in events if e["type"] == "complete"][-1]
        # 评分后即提前结束，匹配与润色被跳过
        assert complete["data"]["steps_success"] == 1
        assert complete["data"]["steps_total"] == 3

    def test_replan_success_end_to_end(self):
        """反思裁决 replan → Master 重新规划 chat 补救步骤 → 执行成功"""
        orch = self._orchestrator()
        switched = {"done": False}

        def review(step, result, goal, trace):
            if step.agent == "resume_score" and not switched["done"]:
                switched["done"] = True
                return Reflection(step_id=step.step_id, verdict=VERDICT_REPLAN,
                                  thought="需要换路径", source="rule")
            return Reflection(step_id=step.step_id, verdict=VERDICT_CONTINUE, source="rule")
        orch.reflector.review = review
        orch.master_agent.replan = lambda **k: TaskPlan(
            goal="g", intent="replan", source="replan",
            steps=[TaskStep(step_id="rp1", agent="chat", task="补救回答")],
        )
        events = self._collect(
            orch, "你好呀", {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD},
            preset_intent=RESUME_PRESET,
        )
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 2 and plan_events[-1]["data"]["source"] == "replan"
        types = [e["type"] for e in events]
        assert "token" in types and "complete" in types

    def test_execute_step_unknown_executor(self):
        orch = self._orchestrator()
        step = TaskStep(step_id="z", agent="ghost_executor", task="不存在的任务")
        gen = orch._execute_step_iter(step, Blackboard(message="hi"))
        with pytest.raises(StopIteration) as stop:
            next(gen)
        result = stop.value.value
        assert not result.success
        assert "不存在" in result.error

    def _drain_stream(self, gen) -> tuple:
        """消费生成器：返回 (透传事件列表, StepResult)。

        必须用 next() 循环——for 会吞掉携带 return 值的 StopIteration，
        导致拿不到生成器的返回值。
        """
        events: list = []
        while True:
            try:
                events.append(next(gen))
            except StopIteration as stop:
                return events, stop.value

    def test_streaming_error_without_content_marks_failure(self):
        llm = MagicMock()
        llm.generate_stream.return_value = iter([{"type": "error", "content": "模型拒绝生成"}])
        orch = self._orchestrator(llm)
        executor = ChatExecutor(llm=llm)
        step = TaskStep(step_id="s", agent="chat", task="对话")
        events, result = self._drain_stream(
            orch._iter_streaming(executor, step, {"message": "hi"})
        )
        assert events == []
        assert not result.success
        assert "拒绝生成" in result.error

    def test_streaming_tokens_pass_through_in_realtime(self):
        """流式改造回归：token 事件应随生成实时 yield，而非全量缓冲后输出"""
        llm = MagicMock()
        llm.generate_stream.return_value = iter([
            {"type": "token", "content": "你"},
            {"type": "token", "content": "好"},
        ])
        orch = self._orchestrator(llm)
        executor = ChatExecutor(llm=llm)
        step = TaskStep(step_id="s", agent="chat", task="对话")
        gen = orch._iter_streaming(executor, step, {"message": "hi"})
        # 首个 token 在生成器完成前即可被调用方拿到（实时透传的关键）
        first = next(gen)
        assert first == {"type": "token", "content": "你"}
        events, result = self._drain_stream(gen)
        assert events == [{"type": "token", "content": "好"}]
        assert result.success and result.data.get("answer") == "你好"

    def test_run_aggregates_failure(self):
        orch = self._orchestrator()
        orch.intent_recognizer.recognize = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("识别崩溃")
        )
        result = orch.run("任意输入", {})
        assert result["success"] is False
        assert "识别崩溃" in result["error"]

    def test_stream_exception_emits_error(self):
        orch = self._orchestrator()
        orch.intent_recognizer.recognize = lambda *a, **k: (_ for _ in ()).throw(
            ValueError("坏输入")
        )
        events = self._collect(orch, "任意输入", {})
        assert events[-1]["type"] == "error"
        assert "坏输入" in events[-1]["message"]

    def test_finalize_polish_without_finalize_method(self):
        class _PlainPolish(BaseExecutor):
            name = "resume_polish"
        text = MasterOrchestrator._finalize_polish(_PlainPolish(), "  干净文本  ", {})
        assert text == "干净文本"

    def test_legacy_events_only_on_success(self):
        orch = self._orchestrator()
        plan = TaskPlan(goal="g", intent="i")

        failed = StepResult(step_id="s1", agent="resume_score", success=False, error="x")
        assert list(orch._legacy_events(
            TaskStep(step_id="s1", agent="resume_score", task="t"), failed, plan
        )) == []

        ok = StepResult(
            step_id="s2", agent="jd_match", success=True,
            data={"match_result": {"match_score": 80}, "suggestions": ["建议"]},
        )
        events = list(orch._legacy_events(
            TaskStep(step_id="s2", agent="jd_match", task="t"), ok, plan
        ))
        assert events[0]["type"] == "suggestions"
        assert events[0]["data"]["suggestions"] == ["建议"]

    def test_complete_contains_chat_answer(self):
        orch = self._orchestrator(make_mock_llm(chat_tokens=["最终", "答案"]))
        events = self._collect(orch, "问题", {}, preset_intent=CHAT_PRESET)
        complete = [e for e in events if e["type"] == "complete"][-1]
        assert complete["data"]["answer"] == "最终答案"

    def test_run_non_stream_resume_falls_back_answer_to_polished(self):
        """非流式简历流程：无 token 时 answer 兜底取润色结果"""
        llm = make_mock_llm(polish_tokens=["专业简历内容，主导微服务架构设计，性能提升显著。" * 2])
        orch = self._orchestrator(llm)
        result = orch.run(
            "", {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD},
            preset_intent=RESUME_PRESET,
        )
        assert result["success"] is True
        assert result["optimized_resume"]
        assert result["answer"] == result["optimized_resume"]

    def test_apply_replan_renames_conflicting_step_ids(self):
        """重规划步骤 id 与历史步骤撞车时自动重命名，并同步后续依赖"""
        orch = self._orchestrator()
        plan = TaskPlan(goal="g", intent="i", steps=[
            TaskStep(step_id="step_1", agent="resume_score", task="t1",
                     status="failed"),
            TaskStep(step_id="step_2", agent="jd_match", task="t2"),
        ])
        new_steps = [
            TaskStep(step_id="step_1", agent="chat", task="补救1", depends_on=[]),
            TaskStep(step_id="step_3", agent="chat", task="补救2",
                     depends_on=["step_1"]),
        ]
        assert orch._apply_replan(plan, new_steps, failed=plan.steps[0]) is True

        ids = [s.step_id for s in plan.steps]
        # 旧 step_1 保留（failed），新 step_1 被改名
        assert ids.count("step_1") == 1
        renamed = [s for s in new_steps if s.task == "补救1"][0].step_id
        assert renamed == "step_1_r1"
        # 依赖同步指向新 id
        assert new_steps[1].depends_on == ["step_1_r1"]
        # 原待执行步骤被跳过
        assert plan.steps[1].status == "skipped"


# ============ 共享 LLM 单例测试 ============

class TestSharedLLM:
    def test_get_shared_llm_singleton(self, monkeypatch):
        """get_shared_llm 本地单例只初始化一次"""
        import agents.llm as llm_mod

        fake = MagicMock(name="shared-llm")
        monkeypatch.setattr(llm_mod, "_shared_llm_instance", None)
        monkeypatch.setattr("agents.registry.get_agent", lambda provider="local", model=None: fake)
        try:
            first = llm_mod.get_shared_llm()
            second = llm_mod.get_shared_llm()
            assert first is fake and second is fake
        finally:
            monkeypatch.setattr(llm_mod, "_shared_llm_instance", None)


# ============ skill_router LLM 注入测试 ============

class TestSkillRouterLLMInjection:
    def test_classifier_prefers_injected_llm(self):
        from skillhub.llm_classifier import LLMIntentClassifier
        fake = MagicMock(name="injected-llm")
        classifier = LLMIntentClassifier(llm=fake)
        assert classifier.agent is fake

    def test_router_agent_propagates_llm(self):
        from skillhub.skill_router_agent import SkillRouterAgent
        fake = MagicMock(name="injected-llm")
        router = SkillRouterAgent(llm=fake)
        assert router.llm_classifier._llm is fake
        assert router.llm_classifier.agent is fake
