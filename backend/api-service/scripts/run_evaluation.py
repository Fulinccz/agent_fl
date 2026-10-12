"""Agent 评测 CLI

用法（在 backend/api-service 目录下）：
    # 本地模型单跑（仅规则评分，不调用裁判 LLM，零成本）
    python scripts/run_evaluation.py --providers local --no-judge

    # 本地 vs DeepSeek 双模型对比 + DeepSeek 当裁判
    python scripts/run_evaluation.py --providers local,deepseek --judge deepseek

    # 保存基线 / 回归检查（指标退化超阈值则退出码非 0，可接 CI）
    python scripts/run_evaluation.py --providers local --save-baseline data/eval_baseline.json
    python scripts/run_evaluation.py --providers local --check-baseline data/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 保证从 scripts/ 直接运行时能 import 顶层包
API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from evaluation.dataset import EvalDataset
from evaluation.runner import run_evaluation, save_report, compare_to_baseline


def _print_markdown(report: dict) -> None:
    print("\n## Agent 评测对比报告\n")
    print(f"用例数：{report['case_count']}    生成时间：{report['generated_at']}\n")
    print("| Provider | 准确率 | 幻觉率 | 相关性 | 规则通过率 | 平均延迟(s) | 平均估算token | 裁判用例数 | 成功/失败 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _name, p in report["providers"].items():
        print(
            f"| {p['provider']} | {p['accuracy']:.3f} | {p['hallucination_rate']:.3f} "
            f"| {p['relevance']:.3f} | {p['rule_pass_rate']:.3f} | {p['avg_latency_s']:.2f} "
            f"| {p['avg_estimated_tokens']:.0f} | {p['judged_n']}/{p['n']} "
            f"| {p['success_n']}/{p['error_n']} |"
        )
    print("\n指标说明：准确率/相关性/规则通过率越高越好；幻觉率越低越好；"
          "平均延迟与 token 仅统计成功用例。\n")

    if report.get("warnings"):
        print("### ⚠️ 告警")
        for w in report["warnings"]:
            print(f"- {w}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 评测与双模型对比")
    parser.add_argument("--providers", default="local",
                        help="逗号分隔的 Provider 列表，如 local,deepseek")
    parser.add_argument("--dataset", default=None, help="评测集 JSON 路径（默认内置）")
    parser.add_argument("--category", default=None,
                        choices=["resume_score", "jd_match", "resume_polish"],
                        help="只评测指定类别")
    parser.add_argument("--judge", default="deepseek", help="裁判模型 Provider（建议与被测不同）")
    parser.add_argument("--judge-model", default=None, help="裁判模型名称（默认用该 Provider 的默认模型）")
    parser.add_argument("--no-judge", action="store_true", help="仅规则评分，不调用裁判 LLM")
    parser.add_argument("--output", default="data/evaluation/eval_report.json", help="报告输出路径")
    parser.add_argument("--save-baseline", default=None, help="将本次报告保存为基线")
    parser.add_argument("--check-baseline", default=None, help="与基线对比，退化则失败")
    args = parser.parse_args()

    dataset = EvalDataset.from_json(args.dataset).filter(args.category)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    report = run_evaluation(
        dataset=dataset,
        providers=providers,
        judge_provider=args.judge,
        judge_model=args.judge_model,
        enable_judge=not args.no_judge,
    )

    save_report(report, args.output)
    print(f"报告已保存：{args.output}")
    _print_markdown(report)

    if args.save_baseline:
        save_report(report, args.save_baseline)
        print(f"基线已保存：{args.save_baseline}")

    if args.check_baseline:
        result = compare_to_baseline(report, args.check_baseline)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            print("\n❌ 评测回归：检测到指标退化超过阈值")
            return 1
        print("\n✅ 评测通过：指标无显著退化")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
