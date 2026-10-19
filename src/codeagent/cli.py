"""命令行入口。

用法：
    python -m codeagent.cli analyze --repo samples/buggy_pkg --no-llm
    python -m codeagent.cli analyze --repo https://github.com/user/repo --max-targets 6
    python -m codeagent.cli serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .schemas import AnalyzeRequest


def _print_summary(report) -> None:
    st = report.stats
    print("=" * 78)
    print(f"仓库：{st.repo}  commit={st.commit or '-'}")
    print(f"规模：{st.files} 文件 / {st.functions} 函数 / {st.classes} 类 / {st.loc} 行"
          f"；检索块 {st.chunks}；解析失败 {len(st.parse_errors)}")
    print(f"LLM：{'开' if report.llm_enabled else '关'}（调用 {report.llm_calls} 次）"
          f"  总耗时 {report.timings_ms.get('wall', 0)/1000:.1f}s")
    print(f"候选缺陷 {report.candidates_total} 条 → 复核后 {len(report.findings)} 条")

    if report.findings:
        print("\n-- 缺陷（按严重度） " + "-" * 56)
        for f in report.findings:
            print(f"[{f.severity.value:8s}] {f.file}:{f.lineno}  {f.title}")
            print(f"           {f.detail[:110]}")
            if f.suggestion:
                print(f"           建议：{f.suggestion[:110]}")

    print("\n-- 待审目标 " + "-" * 62)
    for t in report.targets:
        print(f"  {t.risk_score:.2f}  {t.qualname:32s} {t.file}")
        print(f"        理由：{'；'.join(t.reasons) or '默认'}")

    print("\n-- 测试生成与自检 " + "-" * 56)
    total = sum(len(m.cases) for m in report.tests)
    print(f"  文件 {len(report.tests)} 个 / 用例 {total} 条；自检：{'通过' if report.check.ok else '未通过'}")
    print(f"  有效用例 {report.check.valid_tests}/{report.check.checked_tests}"
          f"；目标覆盖率 {report.check.coverage_ratio:.0%}")
    for issue in report.check.issues:
        if issue.level == "error":
            print(f"  [error] {issue.code}: {issue.message}")

    if report.notes:
        print("\n-- 说明 " + "-" * 68)
        for n in report.notes:
            print(f"  - {n}")
    print("=" * 78)


def cmd_analyze(args: argparse.Namespace) -> int:
    req = AnalyzeRequest(
        repo=args.repo,
        ref=args.ref,
        subdir=args.subdir,
        max_targets=args.max_targets,
        use_llm=not args.no_llm,
        generate_tests=not args.no_tests,
        max_repair=args.max_repair,
    )
    job_id = args.job_id or uuid.uuid4().hex[:8]
    from .agent.graph import run_analysis
    from .repo.loader import RepoLoadError

    try:
        report = run_analysis(req, job_id)
    except RepoLoadError as exc:
        print(f"[仓库准备失败] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        _print_summary(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n报告已写入 {args.out}")
    return 0 if report.check.ok or not req.generate_tests else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("codeagent.api.main:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codeagent", description="代码仓库静态分析 Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="分析一个仓库")
    a.add_argument("--repo", required=True, help="Git URL 或本地路径")
    a.add_argument("--ref", default=None, help="分支/commit（可选）")
    a.add_argument("--subdir", default=None, help="只分析子目录")
    a.add_argument("--max-targets", type=int, default=6)
    a.add_argument("--max-repair", type=int, default=2)
    a.add_argument("--no-llm", action="store_true", help="纯规则模式（不调用模型）")
    a.add_argument("--no-tests", action="store_true", help="跳过测试生成")
    a.add_argument("--json", action="store_true", help="输出完整 JSON")
    a.add_argument("--out", default=None, help="报告写入文件")
    a.add_argument("--job-id", default=None)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
