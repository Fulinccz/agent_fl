"""Agent 评测框架单元测试（mock 被测模型与裁判模型，不发起真实 LLM 调用）"""
import json
import pytest
from unittest import mock

from evaluation.dataset import EvalDataset, EvalCase, GroundTruth, build_prompt
from evaluation.rules import evaluate_rules
from evaluation.judge import _parse_json, judge_output
from evaluation import runner
from evaluation.runner import run_evaluation, compare_to_baseline, save_report, estimate_tokens


# ---------------- dataset ----------------

class TestDataset:
    def test_load_builtin(self):
        ds = EvalDataset.from_json()
        assert len(ds) == 12
        cats = {c.category for c in ds.cases}
        assert {"resume_score", "jd_match", "resume_polish"} <= cats

    def test_filter_category(self):
        ds = EvalDataset.from_json().filter("jd_match")
        assert len(ds) == 4
        assert all(c.category == "jd_match" for c in ds.cases)

    def test_build_prompt_match_includes_jd(self):
        case = EvalCase(id="x", category="jd_match", input="简历", jd="岗位要求Java")
        prompt = build_prompt(case)
        assert "Java" in prompt and "岗位 JD" in prompt

    def test_invalid_category(self):
        with pytest.raises(ValueError):
            EvalCase.from_dict({"id": "1", "category": "nope", "input": "x"})


# ---------------- rules ----------------

class TestRules:
    def test_full_hit(self):
        gt = GroundTruth(must_contain=["Java", "Redis"], structure=["技能", "项目"])
        r = evaluate_rules("技能：Java、Redis；项目经历：订单系统", gt)
        assert r.keyword_coverage == 1.0
        assert r.structure_score == 1.0
        assert r.forbidden_hits == []
        assert r.score == pytest.approx(1.0)

    def test_partial_keyword(self):
        gt = GroundTruth(must_contain=["Java", "K8s"])
        r = evaluate_rules("我会 Java", gt)
        assert r.keyword_coverage == 0.5
        # 无 structure 要求时不再白送结构分：关键词 0.5 独占语义权重 0.9 + 长度 0.1 = 0.55
        assert r.score == pytest.approx(0.55)

    def test_empty_semantic_dimension_not_given_free(self):
        """只有长度约束、无任何语义要求时，不因空维度白送语义分"""
        gt = GroundTruth(min_length=3)
        r = evaluate_rules("合格长度文本", gt)
        assert r.length_ok is True
        assert r.score == pytest.approx(1.0)  # 长度达标即满分（唯一考核维度归一化）
        short = evaluate_rules("短", gt)
        assert short.score == pytest.approx(0.0)  # 长度不达标即 0

    def test_forbidden_penalty(self):
        gt = GroundTruth(forbidden=["保证录用"])
        r = evaluate_rules("保证录用", gt)
        assert r.forbidden_hits == ["保证录用"]
        # 结构/关键词无要求时基础分为 1.0，禁忌词 -0.1
        assert r.score == pytest.approx(0.9)

    def test_length_check(self):
        gt = GroundTruth(min_length=100)
        r = evaluate_rules("短", gt)
        assert r.length_ok is False


# ---------------- judge json parsing ----------------

class TestJudgeParse:
    def test_fenced_json(self):
        raw = '前缀\n```json\n{"correctness": 4, "relevance": 3, "has_hallucination": true}\n```'
        d = _parse_json(raw)
        assert d["correctness"] == 4 and d["has_hallucination"] is True

    def test_bare_json(self):
        d = _parse_json('{"correctness": 2, "relevance": 5, "has_hallucination": false}')
        assert d["relevance"] == 5

    def test_invalid(self):
        assert _parse_json("完全不是JSON") is None
        assert _parse_json("") is None


class TestJudgeOutput:
    def test_judge_success(self):
        case = EvalCase(id="1", category="resume_polish", input="简历")

        def judge_fn(p):
            return '```json\n{"correctness":4,"relevance":4,"has_hallucination":false,"reason":"良好"}\n```'

        s = judge_output("输出", case, judge_fn=judge_fn)
        assert s.available and s.correctness == 4 and s.relevance == 4
        assert s.has_hallucination is False

    def test_judge_fallback_on_garbage(self):
        case = EvalCase(id="1", category="resume_polish", input="简历")
        s = judge_output("输出", case, judge_fn=lambda p: "无法解析的文本")
        assert s.available is False


# ---------------- runner ----------------

def _make_case(cid="c1"):
    return EvalCase(
        id=cid, category="resume_polish", input="候选人会 Java",
        ground_truth=GroundTruth(must_contain=["Java"], structure=["技能"], min_length=3),
    )


class TestRunner:
    def test_aggregates_metrics(self):
        ds = EvalDataset([_make_case("a"), _make_case("b")])
        # 规则全命中输出；裁判固定 4 分、无幻觉
        def judge_fn(p):
            return '{"correctness":4,"relevance":4,"has_hallucination":false,"reason":"ok"}'

        fns = {"local": lambda prompt: "技能：Java 项目经历很丰富" * 2}

        report = run_evaluation(ds, ["local"], generate_fns=fns, judge_fn=judge_fn)
        p = report["providers"]["local"]
        assert p["n"] == 2 and p["judged_n"] == 2
        assert p["hallucination_rate"] == 0.0
        # 规则 1.0 * 0.5 + 正确性 0.8 * 0.5 = 0.9
        assert p["accuracy"] == pytest.approx(0.9, abs=1e-3)
        assert p["relevance"] == pytest.approx(0.8, abs=1e-3)

    def test_hallucination_rate(self):
        ds = EvalDataset([_make_case("a"), _make_case("b")])
        verdicts = iter([
            '{"correctness":2,"relevance":2,"has_hallucination":true,"reason":"编造"}',
            '{"correctness":4,"relevance":4,"has_hallucination":false,"reason":"ok"}',
        ])
        fns = {"local": lambda prompt: "技能：Java 项目"}
        report = run_evaluation(ds, ["local"], generate_fns=fns, judge_fn=lambda p: next(verdicts))
        assert report["providers"]["local"]["hallucination_rate"] == pytest.approx(0.5)

    def test_no_judge_uses_rules_only(self):
        ds = EvalDataset([_make_case("a")])
        fns = {"local": lambda prompt: "技能：Java 项目"}
        report = run_evaluation(ds, ["local"], generate_fns=fns, enable_judge=False)
        p = report["providers"]["local"]
        assert p["judged_n"] == 0
        assert p["accuracy"] == pytest.approx(1.0, abs=1e-3)  # 仅规则，全命中
        assert p["hallucination_rate"] == 0.0

    def test_provider_error_recorded_not_raised(self):
        ds = EvalDataset([_make_case("a")])
        def boom(prompt):
            raise RuntimeError("model down")
        report = run_evaluation(ds, ["local"], generate_fns={"local": boom}, enable_judge=False)
        rec = report["cases"][0]
        assert rec["error"] is not None

    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("a" * 15) == 10


# ---------------- baseline regression ----------------

class TestBaseline:
    def test_detect_accuracy_regression(self, tmp_path):
        baseline = {"providers": {"local": {"accuracy": 0.9, "hallucination_rate": 0.0}}}
        current = {"providers": {"local": {"accuracy": 0.80, "hallucination_rate": 0.0}}}
        bp = tmp_path / "base.json"
        bp.write_text(json.dumps(baseline), encoding="utf-8")
        result = compare_to_baseline(current, bp, accuracy_threshold=0.05)
        assert result["passed"] is False
        assert result["regressions"][0]["metric"] == "accuracy"

    def test_pass_within_threshold(self, tmp_path):
        baseline = {"providers": {"local": {"accuracy": 0.9, "hallucination_rate": 0.0}}}
        current = {"providers": {"local": {"accuracy": 0.88, "hallucination_rate": 0.02}}}
        bp = tmp_path / "base.json"
        bp.write_text(json.dumps(baseline), encoding="utf-8")
        result = compare_to_baseline(current, bp, accuracy_threshold=0.05)
        assert result["passed"] is True

    def test_save_and_load_report(self, tmp_path):
        report = {"providers": {"local": {"accuracy": 0.5}}, "cases": []}
        p = tmp_path / "r.json"
        save_report(report, p)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["providers"]["local"]["accuracy"] == 0.5


# ---------------- 工程保障：缓存 / 熔断 / 告警 ----------------

class TestAgentCaching:
    def test_tested_agent_loaded_only_once(self):
        """本地模型昂贵：同一 provider 多条用例只 get_agent 一次、generate 多次"""
        fake_agent = mock.Mock()
        fake_agent.generate.return_value = "技能：Java 项目经历很丰富"
        fake_get = mock.Mock(return_value=fake_agent)

        cache = {}
        gen = runner._cached_generate_fn("local", None, cache)
        with mock.patch("agents.registry.get_agent", fake_get):
            gen("prompt-1")
            gen("prompt-2")
            gen("prompt-3")

        assert fake_get.call_count == 1
        assert fake_agent.generate.call_count == 3


class TestJudgeCircuitBreaker:
    def test_breaker_stops_calling_judge(self):
        """裁判连续失败达阈值后熔断，剩余用例不再调用裁判"""
        ds = EvalDataset([_make_case("a"), _make_case("b"), _make_case("c")])
        judge_invocations = []

        def garbage_judge(prompt):
            judge_invocations.append(1)
            return "完全无法解析的文本"

        fns = {"local": lambda prompt: "技能：Java 项目"}
        report = run_evaluation(
            ds, ["local"], generate_fns=fns,
            judge_fn=garbage_judge, judge_fail_threshold=2,
        )
        p = report["providers"]["local"]
        assert p["judge_circuit_broken"] is True
        assert p["judged_n"] == 0
        # 每条用例裁判内部重试 2 次；前 2 条触发熔断（4 次调用），第 3 条不再调用（否则 6 次）
        assert len(judge_invocations) == 4

    def test_breaker_is_shared_across_providers(self):
        """裁判服务整体不可用时，熔断跨 Provider 生效，第二个 Provider 不再发起裁判调用"""
        ds = EvalDataset([_make_case("a"), _make_case("b")])
        calls = []

        def garbage_judge(prompt):
            calls.append(1)
            return "无法解析"

        fns = {
            "p1": lambda prompt: "技能：Java 项目",
            "p2": lambda prompt: "技能：React 项目",
        }
        report = run_evaluation(
            ds, ["p1", "p2"], generate_fns=fns,
            judge_fn=garbage_judge, judge_fail_threshold=2,
        )
        # p1 两条各重试 2 次=4；熔断后 p2 不再调用
        assert len(calls) == 4
        assert report["providers"]["p1"]["judge_circuit_broken"] is True
        assert report["providers"]["p2"]["judge_circuit_broken"] is True
        assert report["providers"]["p2"]["judged_n"] == 0


class TestLatencyTokenAccounting:
    def test_failed_cases_excluded_from_latency_and_token_average(self):
        """失败用例的 0 延迟/0 token 不得计入平均，只统计成功用例；但准确率含失败惩罚"""
        ds = EvalDataset([_make_case("a"), _make_case("b")])
        success_text = "技能：Java 项目经历，内容足够长以通过校验"
        state = {"i": 0}

        def gen(prompt):
            state["i"] += 1
            if state["i"] == 1:
                return success_text
            raise RuntimeError("model timeout")

        report = run_evaluation(ds, ["local"], generate_fns={"local": gen}, enable_judge=False)
        p = report["providers"]["local"]
        assert p["n"] == 2 and p["success_n"] == 1 and p["error_n"] == 1
        # 分母为成功数 1：若错误地除以 n=2，token 会是一半
        assert p["avg_estimated_tokens"] == estimate_tokens(success_text)


class TestEvaluationWarnings:
    def test_self_judge_warning(self):
        ds = EvalDataset([_make_case("a")])

        def judge_fn(p):
            return '{"correctness":4,"relevance":4,"has_hallucination":false,"reason":"ok"}'

        fns = {"deepseek": lambda prompt: "技能：Java 项目"}
        report = run_evaluation(
            ds, ["deepseek"], generate_fns=fns,
            judge_fn=judge_fn, judge_provider="deepseek",
        )
        assert any("自我评分" in w for w in report["warnings"])

    def test_mixed_judge_basis_warning(self):
        """一个 provider 有裁判分、另一个全 error 无裁判分时，提示口径不一致"""
        ds = EvalDataset([_make_case("a")])
        fns = {
            "good": lambda prompt: "技能：Java 项目",
            "bad": lambda prompt: (_ for _ in ()).throw(RuntimeError("model down")),
        }

        def judge_fn(p):
            return '{"correctness":4,"relevance":4,"has_hallucination":false,"reason":"ok"}'

        report = run_evaluation(ds, ["good", "bad"], generate_fns=fns, judge_fn=judge_fn)
        assert report["providers"]["good"]["judged_n"] == 1
        assert report["providers"]["bad"]["judged_n"] == 0
        assert any("口径" in w for w in report["warnings"])

    def test_report_records_judge_config(self):
        ds = EvalDataset([_make_case("a")])
        fns = {"local": lambda prompt: "技能：Java 项目"}
        report = run_evaluation(ds, ["local"], generate_fns=fns, enable_judge=False)
        assert report["judge"]["enabled"] is False
        assert report["judge"]["provider"] is None
