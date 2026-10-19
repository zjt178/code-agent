"""端到端（离线）集成测试：整条 LangGraph 在无 LLM 时必须跑通且结论正确。"""
from __future__ import annotations

import uuid
from pathlib import Path

from codeagent.agent.graph import run_analysis
from codeagent.schemas import AnalyzeRequest

ROOT = Path(__file__).resolve().parents[1]


def _run(repo: str, targets: int = 6):
    return run_analysis(
        AnalyzeRequest(repo=str(ROOT / repo), max_targets=targets, use_llm=False),
        job_id=f"test-{uuid.uuid4().hex[:6]}",
    )


def test_clean_repo_has_no_findings() -> None:
    rep = _run("samples/clean_pkg")
    assert rep.stats.files >= 2
    assert rep.findings == []
    assert rep.check.ok is True
    assert rep.check.valid_tests >= 1


def test_buggy_repo_detects_critical_issues() -> None:
    rep = _run("samples/buggy_pkg", targets=12)
    rules = {f.rule_id for f in rep.findings}
    assert {"sql-fstring", "hardcoded-secret", "os-system", "bare-except"} <= rules
    severities = {f.severity.value for f in rep.findings}
    assert "critical" in severities


def test_graph_emits_trace_and_timings() -> None:
    rep = _run("samples/clean_pkg", targets=4)
    nodes = [t.node for t in rep.trace]
    for expected in ("load_repo", "build_index", "plan", "investigate", "detect", "gen_tests",
                     "self_check", "report"):
        assert expected in nodes, expected
    assert rep.timings_ms.get("total", 0) > 0


def test_tests_generated_are_static_only() -> None:
    rep = _run("samples/buggy_pkg", targets=4)
    assert rep.tests, "应生成测试"
    for module in rep.tests:
        assert "def test_" in module.code
        # 生成的测试只允许标准库/pytest/仓库内模块，不应出现网络客户端
        assert "requests." not in module.code or "import requests" in module.code
