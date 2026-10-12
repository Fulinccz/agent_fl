"""评测跑批与指标汇总

同一评测集对多个 Provider 跑批，输出：
- accuracy          准确率 = 0.5*规则通过率 + 0.5*Judge正确性归一化（Judge 不可用时仅规则）
- hallucination_rate 幻觉率 = Judge 判定 has_hallucination 的用例占比（越低越好）
- relevance         相关性 = Judge 相关性均分归一化
- rule_pass_rate    规则通过率均值
- avg_latency_s     平均延迟
- avg_estimated_tokens 平均估算 token（中文粗估 len/1.5，用于成本对比）

工程保障：
- 被测 / 裁判 Provider 均按 (provider, model) 缓存复用，本地 4B 模型全程只加载一次
- 裁判直连 registry，绕过线上 Provider 成本开关，避免"未启用静默回落 local"导致对比失真
- 裁判熔断：连续 N 次不可用则本 Provider 后续跳过裁判，避免无 Key 时每条重试拖慢跑批
- self-judge 检测与评分口径告警写入报告 warnings

基线回归：compare_to_baseline 对比已保存报告，指标退化超阈值即返回失败。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any

from logger import get_logger
from .dataset import EvalDataset, build_prompt
from .rules import evaluate_rules
from .judge import JudgeScore, judge_output, JudgeFn

logger = get_logger(__name__)

# Generate 函数：(prompt) -> 输出文本
GenerateFn = Callable[[str], str]


@dataclass
class ProviderReport:
    provider: str
    n: int = 0
    success_n: int = 0
    error_n: int = 0
    judged_n: int = 0
    accuracy: float = 0.0
    hallucination_rate: float = 0.0
    relevance: float = 0.0
    rule_pass_rate: float = 0.0
    avg_latency_s: float = 0.0
    avg_estimated_tokens: float = 0.0
    judge_circuit_broken: bool = False   # 裁判是否因连续失败被熔断

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def estimate_tokens(text: str) -> int:
    """估算 token 数（中文约 1.5 字符/token 的粗估，仅用于成本量级对比）"""
    return int(len(text or "") / 1.5)


def _cached_generate_fn(provider: str, model: Optional[str], cache: Dict[tuple, Any]) -> GenerateFn:
    """构造复用缓存 Agent 的生成函数（本地模型全程只加载一次）"""
    def _call(prompt: str) -> str:
        key = (provider, model)
        if key not in cache:
            # 评测为显式离线跑批，直连 registry 绕过成本开关，杜绝静默回落 local
            from agents.registry import get_agent
            logger.info("评测加载被测 Agent：provider=%s model=%s", provider, model)
            cache[key] = get_agent(provider=provider, model=model)
        return cache[key].generate(prompt)
    return _call


def run_evaluation(
    dataset: EvalDataset,
    providers: List[str],
    generate_fns: Optional[Dict[str, GenerateFn]] = None,
    judge_fn: Optional[JudgeFn] = None,
    judge_provider: str = "deepseek",
    judge_model: Optional[str] = None,
    enable_judge: bool = True,
    model: Optional[str] = None,
    judge_fail_threshold: int = 2,
) -> Dict[str, Any]:
    """对多个 Provider 跑同一评测集

    Args:
        providers: 被测 Provider 列表
        generate_fns: 可注入 {provider: 生成函数}，测试传 mock；缺省走 registry（缓存复用）
        judge_fn: 可注入裁判函数；缺省用 judge_provider/judge_model（与被测模型独立配置）
        judge_model: 裁判模型（与被测 model 分离）
        model: 被测模型（可选）
        judge_fail_threshold: 裁判连续不可用多少次后熔断
    """
    generate_fns = generate_fns or {}
    tested_agent_cache: Dict[tuple, Any] = {}
    case_records: List[Dict[str, Any]] = []
    provider_reports: Dict[str, ProviderReport] = {}
    warnings: List[str] = []

    # self-judge 偏差检测
    if enable_judge and judge_provider in providers:
        msg = (f"裁判模型（{judge_provider}）同时是被测模型，存在自我评分偏差，"
               f"建议用独立裁判：--judge <其它 provider>")
        warnings.append(msg)
        logger.warning(msg)

    # 裁判熔断为全局状态：裁判服务本身不可用（如无 API Key）时所有 Provider 都会失败，
    # 一旦熔断，后续 Provider 不再重复重试，直接降级纯规则
    judge_state = {"streak": 0, "broken": False}

    for provider in providers:
        gen = generate_fns.get(provider) or _cached_generate_fn(provider, model, tested_agent_cache)
        agg = {"rule": 0.0, "correct": 0.0, "relevance": 0.0, "hallu": 0,
               "judged": 0, "latency": 0.0, "tokens": 0, "n": 0, "success": 0,
               "errors": 0, "accuracy": 0.0}
        judge_degraded = False

        for case in dataset.cases:
            prompt = build_prompt(case)
            output = ""
            latency = 0.0
            error: Optional[str] = None
            try:
                t0 = time.perf_counter()
                output = gen(prompt) or ""
                latency = time.perf_counter() - t0
            except Exception as e:
                error = str(e)
                logger.error("Provider=%s 用例=%s 执行失败：%s", provider, case.id, e)

            rule = evaluate_rules(output, case.ground_truth)

            judge_result = JudgeScore(available=False)
            if enable_judge and not error and not judge_state["broken"]:
                judge_result = judge_output(
                    output, case, judge_fn=judge_fn,
                    judge_provider=judge_provider, judge_model=judge_model,
                )
                if judge_result.available:
                    judge_state["streak"] = 0
                else:
                    judge_state["streak"] += 1
                    if judge_state["streak"] >= judge_fail_threshold:
                        judge_state["broken"] = True
                        judge_degraded = True
                        warnings.append(
                            f"裁判连续 {judge_fail_threshold} 次不可用，已全局熔断，"
                            f"剩余用例（含其它 Provider）降级为纯规则评分；"
                            f"准确率口径与完整裁判结果不可直接比较"
                        )
                        logger.warning("裁判连续失败达阈值，全局熔断后续裁判调用")
            elif enable_judge and judge_state["broken"]:
                judge_degraded = True

            # 单条准确率：裁判可用=规则0.5+裁判0.5；不可用=纯规则
            if judge_result.available:
                accuracy = rule.score * 0.5 + (judge_result.correctness / 5.0) * 0.5
            else:
                accuracy = rule.score

            agg["n"] += 1
            agg["rule"] += rule.score
            agg["accuracy"] += accuracy
            if error:
                agg["errors"] += 1
            else:
                # 延迟与 token 仅统计成功用例：失败用例 latency=0 若计入会把平均延迟拉低，
                # 造成"越超时性能指标越好"的假象
                agg["success"] += 1
                agg["latency"] += latency
                agg["tokens"] += estimate_tokens(output)
            if judge_result.available:
                agg["judged"] += 1
                agg["correct"] += judge_result.correctness / 5.0
                agg["relevance"] += judge_result.relevance / 5.0
                if judge_result.has_hallucination:
                    agg["hallu"] += 1

            case_records.append({
                "case_id": case.id,
                "category": case.category,
                "provider": provider,
                "error": error,
                "latency_s": round(latency, 3),
                "estimated_tokens": estimate_tokens(output),
                "accuracy": round(accuracy, 4),
                "rule": rule.to_dict(),
                "judge": judge_result.to_dict(),
                "output_excerpt": output[:300],
            })

        n = agg["n"] or 1
        success = agg["success"]
        judged = agg["judged"]
        provider_reports[provider] = ProviderReport(
            provider=provider,
            n=agg["n"],
            success_n=success,
            error_n=agg["errors"],
            judged_n=judged,
            accuracy=agg["accuracy"] / n,
            hallucination_rate=(agg["hallu"] / judged) if judged else 0.0,
            relevance=(agg["relevance"] / judged) if judged else 0.0,
            rule_pass_rate=agg["rule"] / n,
            avg_latency_s=(agg["latency"] / success) if success else 0.0,
            avg_estimated_tokens=(agg["tokens"] / success) if success else 0.0,
            judge_circuit_broken=judge_degraded,
        )

    # 跨 Provider 评分口径不一致告警
    judged_counts = [r.judged_n for r in provider_reports.values()]
    if enable_judge and judged_counts and any(c > 0 for c in judged_counts) \
            and any(c == 0 for c in judged_counts):
        warnings.append(
            "部分 Provider 完全没有裁判评分（准确率=纯规则），其余为规则+裁判混合口径，"
            "跨模型准确率不可直接比较，建议修复裁判后重跑"
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(dataset.cases),
        "judge": {
            "enabled": enable_judge,
            "provider": judge_provider if enable_judge else None,
            "model": judge_model,
        },
        "warnings": warnings,
        "providers": {name: r.to_dict() for name, r in provider_reports.items()},
        "cases": case_records,
    }


def save_report(report: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_report(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_to_baseline(
    report: Dict[str, Any],
    baseline_path: str | Path,
    accuracy_threshold: float = 0.05,
    hallucination_threshold: float = 0.1,
) -> Dict[str, Any]:
    """与基线报告对比，检测指标退化

    Returns:
        {"passed": bool, "regressions": [...], "compared": [...]}
    """
    baseline = load_report(baseline_path)
    regressions: List[Dict[str, Any]] = []
    compared: List[Dict[str, Any]] = []

    for provider, cur in report.get("providers", {}).items():
        base = baseline.get("providers", {}).get(provider)
        if not base:
            continue
        acc_drop = base.get("accuracy", 0) - cur.get("accuracy", 0)
        hallu_rise = cur.get("hallucination_rate", 0) - base.get("hallucination_rate", 0)

        compared.append({
            "provider": provider,
            "accuracy_delta": round(cur.get("accuracy", 0) - base.get("accuracy", 0), 4),
            "hallucination_delta": round(cur.get("hallucination_rate", 0) - base.get("hallucination_rate", 0), 4),
        })
        if acc_drop > accuracy_threshold:
            regressions.append({
                "provider": provider, "metric": "accuracy",
                "drop": round(acc_drop, 4), "threshold": accuracy_threshold,
            })
        if hallu_rise > hallucination_threshold:
            regressions.append({
                "provider": provider, "metric": "hallucination_rate",
                "rise": round(hallu_rise, 4), "threshold": hallucination_threshold,
            })

    return {"passed": not regressions, "regressions": regressions, "compared": compared}
