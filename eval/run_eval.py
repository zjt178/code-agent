"""评测：召回率 / 额外发现 / 误报基线 / 测试自检通过率。

设计：
- 召回：在标注集（20 条注入缺陷）上算 matched / total；
- 误报基线：在 clean_pkg（人工确认无缺陷、且规则零命中）上跑同一条流水线，
  任何结论都记为误报——这是最严格的"精确率代理指标"；
- 额外发现：buggy_pkg 上未标注但被报出的条目，单独列出，不直接算误报（需人工判定）。

用法：
    python eval/run_eval.py            # 纯规则模式（确定性、可复现）
    python eval/run_eval.py --llm      # 带模型复核（会慢很多）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeagent.agent.graph import run_analysis  # noqa: E402
from codeagent.schemas import AnalyzeRequest  # noqa: E402

LABELS = json.loads((ROOT / "eval/dataset/buggy_labels.json").read_text(encoding="utf-8"))["labels"]


def _key(symbol_id: str, rule_id: str) -> str:
    return f"{symbol_id}|{rule_id}"


def evaluate(use_llm: bool, repeat: int = 1, max_targets: int = 12) -> dict:
    label_keys = {_key(x["symbol_id"], x["rule_id"]) for x in LABELS}
    runs = []
    for i in range(repeat):
        t0 = time.time()
        rep = run_analysis(
            AnalyzeRequest(repo=str(ROOT / "samples/buggy_pkg"), max_targets=max_targets, use_llm=use_llm),
            job_id=f"eval-buggy-{uuid.uuid4().hex[:6]}",
        )
        found = {_key(f.symbol_id, f.rule_id) for f in rep.findings}
        matched = sorted(label_keys & found)
        extra = sorted(found - label_keys)
        miss = sorted(label_keys - found)
        runs.append({
            "elapsed_s": round(time.time() - t0, 1),
            "recall": len(matched) / len(label_keys),
            "matched": matched, "missing": miss, "extra": extra,
            "findings": len(rep.findings),
            "candidates": rep.candidates_total,
            "valid_tests": rep.check.valid_tests,
            "checked_tests": rep.check.checked_tests,
            "coverage": round(rep.check.coverage_ratio, 3),
            "llm_calls": rep.llm_calls,
            "repairs": sum(1 for t in rep.trace if t.node == "repair"),
        })
        print(f"  run{i + 1}: recall {runs[-1]['recall']:.1%}  结论 {runs[-1]['findings']}  "
              f"额外 {len(extra)}  用时 {runs[-1]['elapsed_s']}s")

    clean = run_analysis(
        AnalyzeRequest(repo=str(ROOT / "samples/clean_pkg"), max_targets=8, use_llm=use_llm),
        job_id=f"eval-clean-{uuid.uuid4().hex[:6]}",
    )
    clean_findings = [f"{f.file}:{f.lineno} {f.rule_id}" for f in clean.findings]

    summary = {
        "mode": "llm" if use_llm else "rule-only",
        "labels": len(label_keys),
        "recall_mean": round(statistics.mean(r["recall"] for r in runs), 3),
        "findings_mean": round(statistics.mean(r["findings"] for r in runs), 2),
        "extra_total": sorted({e for r in runs for e in r["extra"]}),
        "missing_total": sorted({e for r in runs for e in r["missing"]}),
        "clean_false_positives": clean_findings,
        "test_valid_rate": round(runs[0]["valid_tests"] / runs[0]["checked_tests"], 3) if runs[0]["checked_tests"] else None,
        "test_coverage": runs[0]["coverage"],
        "llm_calls": runs[0]["llm_calls"],
        "runs": runs,
    }
    out_dir = ROOT / "eval/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{summary['mode']}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="启用模型复核")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-targets", type=int, default=12)
    args = ap.parse_args()

    print(f"== 评测（{'LLM 复核' if args.llm else '纯规则'}）==")
    s = evaluate(args.llm, args.repeat, args.max_targets)
    print("\n==== 结果 ====")
    print(f"标注缺陷数      : {s['labels']}")
    print(f"召回率          : {s['recall_mean']:.1%}")
    print(f"结论条数        : {s['findings_mean']}")
    print(f"额外发现        : {len(s['extra_total'])} 条（未标注，需人工判定）")
    for e in s["extra_total"]:
        print(f"   - {e}")
    if s["missing_total"]:
        print(f"漏报            : {len(s['missing_total'])} 条")
        for m in s["missing_total"]:
            print(f"   - {m}")
    print(f"clean_pkg 误报  : {len(s['clean_false_positives'])} 条 {s['clean_false_positives']}")
    print(f"测试用例有效率  : {s['test_valid_rate']}")
    print(f"目标覆盖率      : {s['test_coverage']:.0%}")
    print(f"LLM 调用        : {s['llm_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
