"""LangGraph 节点实现。

图结构（有向，带一个修复回环）：
    load_repo → build_index → plan → investigate → detect → gen_tests → self_check
                                                     ↑                      │
                                                     └──── repair ←─────────┘
                                                                           ↓
                                                                        report → END

约定：每个节点只做一件事，返回 state 的增量；所有节点都要能在 LLM 不可用时降级运行。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, TypedDict

from ..analysis.llm import LLMClient
from ..analysis.rules import RULES, run_rules
from ..config import SETTINGS
from ..repo.ast_index import CodeIndex
from ..repo.loader import LoadedRepo, load_repo
from ..retrieval.embed import build_embedder
from ..retrieval.hybrid import HybridRetriever
from ..schemas import (
    AnalyzeRequest, Candidate, CheckReport, Evidence, Finding, RepoStats, Severity, SymbolKind, TargetPlan,
    TestModule, TraceEvent,
)
from ..testing.check import check_all
from ..testing.gen import generate_tests, repair_module
from .prompts import DETECT_SYSTEM, INVESTIGATE_SYSTEM, PLAN_SYSTEM
from .tools import build_tools

SEVERITY_ORDER = {
    Severity.CRITICAL: 5, Severity.HIGH: 4, Severity.MEDIUM: 3, Severity.LOW: 2, Severity.INFO: 1,
}


class AgentState(TypedDict, total=False):
    request: AnalyzeRequest
    job_id: str
    loaded: LoadedRepo
    index: CodeIndex
    retriever: HybridRetriever
    llm: LLMClient
    embed_note: str
    stats: RepoStats
    targets: list[TargetPlan]
    evidence: dict[str, list[Evidence]]
    candidates: list[Candidate]
    findings: list[Finding]
    tests: list[TestModule]
    test_targets: list[str]
    check: CheckReport
    retries: int
    trace: list[TraceEvent]
    timings: dict[str, float]
    notes: list[str]


def _ev(state: AgentState, node: str, action: str, detail: str = "", ms: float = 0.0, **payload: Any) -> None:
    state.setdefault("trace", []).append(
        TraceEvent(node=node, action=action, detail=detail, ms=round(ms, 1), payload=payload)
    )


def _tick(state: AgentState, node: str, ms: float) -> None:
    state.setdefault("timings", {})[node] = round(ms, 1)


# --------------------------------------------------------------------------- 1
def load_repo_node(state: AgentState) -> dict:
    t0 = time.time()
    req = state["request"]
    loaded = load_repo(req.repo, ref=req.ref, subdir=req.subdir,
                       workdir=SETTINGS.storage_dir / state["job_id"])
    _ev(state, "load_repo", "cloned" if loaded.commit else "scanned",
        f"{len(loaded.py_files)} 个 .py 文件", (time.time() - t0) * 1000,
        root=str(loaded.root), truncated=loaded.truncated)
    _tick(state, "load_repo", (time.time() - t0) * 1000)
    notes = list(state.get("notes", []))
    if loaded.truncated:
        notes.append(f"仓库超过分析上限（{SETTINGS.max_files} 文件 / {SETTINGS.max_repo_mb}MB），已截断")
    if loaded.skipped:
        notes.append(f"跳过 {len(loaded.skipped)} 个超大文件")
    return {"loaded": loaded, "notes": notes}


# --------------------------------------------------------------------------- 2
def build_index_node(state: AgentState) -> dict:
    t0 = time.time()
    loaded = state["loaded"]
    index = CodeIndex().build(loaded.root, loaded.rel_files)
    chunks = index.to_chunks()
    embedder, note = build_embedder()
    retriever = HybridRetriever(chunks, embedder, note)

    st = index.stats()
    stats = RepoStats(
        repo=loaded.display, commit=loaded.commit, files=st["files"], symbols=st["symbols"],
        functions=st["functions"], classes=st["classes"], loc=st["loc"], chunks=len(chunks),
        parse_errors=st["parse_errors"],
    )
    ms = (time.time() - t0) * 1000
    _ev(state, "build_index", "indexed",
        f"{stats.files} 文件 / {stats.symbols} 符号 / {stats.chunks} 检索块；{note}", ms,
        embedder=embedder.name, parse_errors=len(stats.parse_errors))
    _tick(state, "build_index", ms)
    notes = list(state.get("notes", []))
    notes.append(note)
    if stats.parse_errors:
        notes.append(f"{len(stats.parse_errors)} 个文件解析失败（语法错误或编码问题），已跳过")
    return {"index": index, "retriever": retriever, "stats": stats, "embed_note": note, "notes": notes}


# --------------------------------------------------------------------------- 3
def _is_test_file(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return (
        any(p in ("tests", "test", "testing", "examples", "docs") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name in ("conftest.py", "setup.py", "noxfile.py")
    )


def _risk(sym, hits: list[Candidate], callers: int) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if sym.complexity > 10:
        score += 0.30
        reasons.append(f"圈复杂度 {sym.complexity}")
    elif sym.complexity > 5:
        score += 0.15
        reasons.append(f"圈复杂度 {sym.complexity}")
    if hits:
        # 严重度加权：安全/可靠性问题即使函数很短也必须进审查队列
        weight = {Severity.CRITICAL: 0.45, Severity.HIGH: 0.25, Severity.MEDIUM: 0.10,
                  Severity.LOW: 0.04, Severity.INFO: 0.0}
        add = min(0.65, sum(weight[h.severity] for h in hits))
        score += add
        worst = max(hits, key=lambda h: SEVERITY_ORDER.get(h.severity, 0))
        reasons.append(f"{len(hits)} 条规则命中（最高 {worst.severity.value}）")
    if not sym.docstring:
        score += 0.05
    if sym.n_lines > 40:
        score += 0.10
        reasons.append(f"{sym.n_lines} 行")
    if callers:
        score += min(0.15, 0.03 * callers)
        reasons.append(f"{callers} 处调用")
    if not sym.name.startswith("_"):
        score += 0.05
    return round(max(score, 0.01), 3), reasons


def plan_node(state: AgentState) -> dict:
    t0 = time.time()
    index: CodeIndex = state["index"]
    req: AnalyzeRequest = state["request"]
    llm: LLMClient = state["llm"]

    all_candidates = run_rules(index)
    by_symbol: dict[str, list[Candidate]] = {}
    for c in all_candidates:
        by_symbol.setdefault(c.symbol_id, []).append(c)

    scored: list[tuple[float, Any, list[str]]] = []
    for sym in index.functions():
        if _is_test_file(sym.file) or sym.n_lines < 3:
            continue
        if sym.qualname.split(".")[-1] in {"__init__", "__repr__", "__str__", "__eq__"} and sym.complexity <= 3:
            continue
        callers = len(index.callers.get(sym.id, []))
        score, reasons = _risk(sym, by_symbol.get(sym.id, []), callers)
        scored.append((score, sym, reasons))
    scored.sort(key=lambda x: x[0], reverse=True)
    shortlist = scored[: max(req.max_targets * 2, 12)]

    picked: list[TargetPlan] = []
    reason_map = {s.id: (score, reasons) for score, s, reasons in shortlist}
    order: list[str] = []

    if llm.enabled and shortlist:
        payload = [
            {"symbol_id": s.id, "file": s.file, "complexity": s.complexity, "lines": s.n_lines,
             "rule_hits": [c.rule_id for c in by_symbol.get(s.id, [])], "score": score}
            for score, s, _ in shortlist
        ]
        out = llm.chat_json(PLAN_SYSTEM, json.dumps(payload, ensure_ascii=False))
        if isinstance(out, list):
            for item in out:
                sid = item.get("symbol_id") if isinstance(item, dict) else None
                if sid in reason_map and sid not in order:
                    order.append(sid)
            _ev(state, "plan", "llm_rerank", f"模型挑选 {len(order)} 个目标", (time.time() - t0) * 1000)

    for sid, (score, reasons) in sorted(reason_map.items(), key=lambda kv: kv[1][0], reverse=True):
        if sid not in order:
            order.append(sid)

    for sid in order[: req.max_targets]:
        sym = index.symbols[sid]
        score, reasons = reason_map[sid]
        picked.append(TargetPlan(symbol_id=sid, file=sym.file, qualname=sym.qualname,
                                 risk_score=score, reasons=reasons))

    ms = (time.time() - t0) * 1000
    _ev(state, "plan", "selected", f"{len(picked)} 个待审函数（共 {len(scored)} 候选）", ms,
        targets=[t.symbol_id for t in picked])
    _tick(state, "plan", ms)
    return {"targets": picked, "candidates": all_candidates}


# --------------------------------------------------------------------------- 4
def investigate_node(state: AgentState) -> dict:
    """为每个目标取证。

    分工：模型决定"查什么、要不要看调用方"，执行层用 LangChain Tool 落地
    （search_code / get_callers）。不让小模型直接做原生 tool-calling——它在工具协议上
    不稳定，会把"该调工具"退化成"直接编答案"；这里只让它输出结构化参数，可校验、可复现。
    """
    t0 = time.time()
    index: CodeIndex = state["index"]
    retriever: HybridRetriever = state["retriever"]
    llm: LLMClient = state["llm"]
    tools = {t.name: t for t in build_tools(index, retriever)}
    evidence: dict[str, list[Evidence]] = {}

    for plan in state.get("targets", []):
        sym = index.get(plan.symbol_id)
        if sym is None:
            continue
        items: list[Evidence] = [
            Evidence(symbol_id=sym.id, kind="self", reason="目标函数本体", snippet=sym.source[:1500])
        ]

        queries: list[str] = []
        inspect_callers = True
        skip = False
        if llm.enabled:
            user = (
                f"目标函数：{sym.qualname}\n文件：{sym.file}:{sym.lineno}\n"
                f"docstring：{sym.docstring or '无'}\n源码：\n```python\n{sym.source[:1600]}\n```\n"
                f"已命中规则：{[c.rule_id for c in state.get('candidates', []) if c.symbol_id == sym.id]}"
            )
            out = llm.chat_json(INVESTIGATE_SYSTEM, user)
            if isinstance(out, dict):
                queries = [q for q in out.get("queries", []) if isinstance(q, str)][:2]
                inspect_callers = bool(out.get("inspect_callers", True))
                skip = bool(out.get("skip", False))
        if not queries:
            queries = [f"{sym.qualname} {sym.docstring or ''}".strip()[:80]]

        for q in queries:
            raw = tools["search_code"].invoke({"query": q, "k": 3})
            try:
                hits = json.loads(raw)
            except json.JSONDecodeError:
                hits = []
            for hit in hits:
                if hit.get("symbol_id") == sym.id:
                    continue
                items.append(Evidence(symbol_id=hit["symbol_id"], kind="similar",
                                      reason=f"检索：{q}", snippet=hit.get("preview", "")[:700],
                                      score=float(hit.get("score", 0.0))))

        if inspect_callers:
            raw = tools["get_callers"].invoke({"symbol_id": sym.id, "limit": 3})
            try:
                for caller in json.loads(raw):
                    items.append(Evidence(symbol_id=caller["symbol_id"], kind="caller",
                                          reason="调用方（影响面）", snippet=caller.get("source", "")[:700]))
            except (json.JSONDecodeError, KeyError):
                pass

        # 去重，控制 prompt 体积
        seen: set[str] = set()
        uniq: list[Evidence] = []
        for e in items:
            key = f"{e.symbol_id}:{e.kind}"
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        evidence[sym.id] = uniq[:7]
        if skip:
            state.setdefault("notes", []).append(f"{sym.qualname} 被模型判定为低价值目标（已跳过深审）")

    ms = (time.time() - t0) * 1000
    _ev(state, "investigate", "evidence",
        f"{len(evidence)} 个目标共收集 {sum(len(v) for v in evidence.values())} 条证据", ms)
    _tick(state, "investigate", ms)
    return {"evidence": evidence}


# --------------------------------------------------------------------------- 5
SEV_BY_NAME = {s.value: s for s in Severity}


def _adjudicate(llm: LLMClient, index: CodeIndex, sym, cands: list[Candidate],
                 evs: list[Evidence]) -> tuple[list[Finding], str]:
    """LLM 复核；返回 (findings, 说明)。无 LLM 时直接把候选转为结论。"""
    if not llm.enabled:
        out = []
        for c in cands:
            out.append(Finding(
                id="", symbol_id=c.symbol_id, file=c.file, lineno=c.lineno, rule_id=c.rule_id,
                category=c.category, severity=c.severity, title=c.title,
                detail=c.message if c.message != c.title else (c.snippet[:160] or c.title),
                suggestion=c.suggestion, confidence=0.55, confirmed_by="rule",
                evidence=[e.reason for e in evs[:2]],
            ))
        return out, "rule-only"

    payload = {
        "file": sym.file,
        "symbol_id": sym.id,
        "source": sym.source[:2600],
        "candidates": [
            {"rule_id": c.rule_id, "lineno": c.lineno, "title": c.title, "message": c.message}
            for c in cands
        ],
        "context": [{"kind": e.kind, "reason": e.reason, "snippet": e.snippet[:500]} for e in evs[:6]],
    }
    data = llm.chat_json(DETECT_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(data, list):
        return _adjudicate_fallback(cands, evs)

    findings: list[Finding] = []
    # 行号只信 AST 的结果：模型自报的 lineno 经常是相对行号或幻觉
    line_by_rule = {c.rule_id: c.lineno for c in cands}
    msg_by_rule = {c.rule_id: (c.message if c.message != c.title else c.snippet[:160]) for c in cands}
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("verdict") is False:
            continue
        rule_id = str(item.get("rule_id") or "llm-new")
        meta = RULES.get(rule_id)
        sev = SEV_BY_NAME.get(str(item.get("severity", "")).lower(), meta.severity if meta else Severity.MEDIUM)
        try:
            conf = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        lineno = int(line_by_rule.get(rule_id) or sym.lineno)
        detail = str(item.get("detail") or "").strip()
        if len(detail) < 8:  # 模型没给出有效证据，退回规则原文
            detail = msg_by_rule.get(rule_id) or (meta.title if meta else "模型发现的问题")
        findings.append(Finding(
            id="", symbol_id=sym.id, file=sym.file, lineno=lineno,
            rule_id=rule_id, category=meta.category if meta else "llm",
            severity=sev, title=str(item.get("title") or (meta.title if meta else "模型发现的问题"))[:60],
            detail=detail, suggestion=str(item.get("suggestion") or ""),
            confidence=min(max(conf, 0.0), 1.0), confirmed_by="llm",
            evidence=[e.reason for e in evs[:3]],
        ))
    return findings, f"llm-judged {len(data)}"


def _adjudicate_fallback(cands: list[Candidate], evs: list[Evidence]) -> tuple[list[Finding], str]:
    return [
        Finding(id="", symbol_id=c.symbol_id, file=c.file, lineno=c.lineno, rule_id=c.rule_id,
                category=c.category, severity=c.severity, title=c.title,
                detail=c.message if c.message != c.title else (c.snippet[:160] or c.title),
                suggestion=c.suggestion, confidence=0.55, confirmed_by="rule",
                evidence=[e.reason for e in evs[:2]])
        for c in cands
    ], "rule-fallback"


def detect_node(state: AgentState) -> dict:
    """规则全仓扫描（便宜）→ 结论全量输出；LLM 复核只用于预算内的目标。

    这样召回由规则保证，精度由复核提升，两者的代价是分开计的：
    candidates 是规则命中数，复核只影响 confirmed_by 与 confidence。
    """
    t0 = time.time()
    index: CodeIndex = state["index"]
    llm: LLMClient = state["llm"]
    targets = state.get("targets", [])
    target_ids = {t.symbol_id for t in targets}

    MAINT_NOISE_IN_TESTS = {"long-function", "missing-docstring", "too-many-params",
                            "deep-nesting", "high-complexity"}
    cands = [c for c in (state.get("candidates") or run_rules(index))
             if not (_is_test_file(c.file) and c.rule_id in MAINT_NOISE_IN_TESTS)]
    by_symbol: dict[str, list[Candidate]] = {}
    for c in cands:
        by_symbol.setdefault(c.symbol_id, []).append(c)

    findings: list[Finding] = []
    judged = 0
    for sid, cs in by_symbol.items():
        sym = index.get(sid)
        if sym is None:
            continue
        if llm.enabled and sid in target_ids:
            evs = state.get("evidence", {}).get(sid, [])
            out, how = _adjudicate(llm, index, sym, cs, evs)
            if how.startswith("llm"):
                judged += 1
            findings.extend(out)
        else:
            evs = state.get("evidence", {}).get(sid, [])
            out, _ = _adjudicate_fallback(cs, evs)
            for f in out:
                f.confidence = 0.5
                if llm.enabled:
                    f.evidence.append("未复核（超出复核预算）")
            findings.extend(out)

    # 去重 + 编号
    seen: set[tuple] = set()
    uniq: list[Finding] = []
    for f in findings:
        key = (f.file, f.rule_id, f.lineno, f.title)
        if key in seen:
            continue
        seen.add(key)
        f.id = f"F{len(uniq) + 1:03d}"
        uniq.append(f)
    uniq.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 0), f.file, f.lineno), reverse=True)

    notes = list(state.get("notes", []))
    info = [f for f in uniq if f.severity == Severity.INFO]
    if info and os.getenv("CODEAGENT_INCLUDE_INFO", "0") != "1":
        uniq = [f for f in uniq if f.severity != Severity.INFO]
        notes.append(f"{len(info)} 条 info 级建议（缺 docstring 等）未列入结论，"
                     f"设置 CODEAGENT_INCLUDE_INFO=1 可查看")

    ms = (time.time() - t0) * 1000
    _ev(state, "detect", "adjudicated",
        f"候选 {len(cands)} 条 → 结论 {len(uniq)} 条（LLM 复核 {judged} 个函数）", ms,
        by_severity={s.value: sum(1 for f in uniq if f.severity == s) for s in Severity})
    _tick(state, "detect", ms)
    return {"findings": uniq, "notes": notes}


# --------------------------------------------------------------------------- 6
def gen_tests_node(state: AgentState) -> dict:
    t0 = time.time()
    req: AnalyzeRequest = state["request"]
    if not req.generate_tests:
        _ev(state, "gen_tests", "skipped", "未启用测试生成")
        return {"tests": []}

    index: CodeIndex = state["index"]
    findings = state.get("findings", [])

    # 测试对象 = 计划审查的函数；高危但不在计划内的函数再补 2 个（预算内加测）
    planned = [
        t.symbol_id for t in state.get("targets", [])
        if index.symbols.get(t.symbol_id) and
        index.symbols[t.symbol_id].kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)
    ]
    extra: list[str] = []
    for f in findings:
        if f.severity in (Severity.CRITICAL, Severity.HIGH) and f.symbol_id not in planned \
                and f.symbol_id not in extra:
            extra.append(f.symbol_id)
    ranked = (planned + extra[:2])[: req.max_targets + 2]

    modules = generate_tests(state["llm"], index, ranked, findings, per_file_limit=6)
    ms = (time.time() - t0) * 1000
    _ev(state, "gen_tests", "generated",
        f"{len(modules)} 个测试文件 / {sum(len(m.cases) for m in modules)} 条用例", ms,
        mode="llm" if state["llm"].enabled else "template", targets=ranked)
    _tick(state, "gen_tests", ms)
    return {"tests": modules, "test_targets": ranked}


# --------------------------------------------------------------------------- 7
def self_check_node(state: AgentState) -> dict:
    t0 = time.time()
    index: CodeIndex = state["index"]
    modules = state.get("tests", [])
    # 覆盖率分母 = 本轮实际打算覆盖的函数（计划目标 + 高危加测）
    planned = state.get("test_targets") or [
        t.symbol_id for t in state.get("targets", [])
        if index.symbols.get(t.symbol_id) and
        index.symbols[t.symbol_id].kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)
    ]
    report = check_all(modules, index, planned)
    ms = (time.time() - t0) * 1000
    _ev(state, "self_check", "checked" if report.ok else "failed",
        f"{report.checked_tests} 条用例，有效 {report.valid_tests}，失败 {len([i for i in report.issues if i.level == 'error'])}",
        ms, coverage=round(report.coverage_ratio, 3))
    _tick(state, "self_check", ms)
    return {"check": report}


# --------------------------------------------------------------------------- 8
def repair_node(state: AgentState) -> dict:
    t0 = time.time()
    index: CodeIndex = state["index"]
    check: CheckReport = state["check"]
    modules: list[TestModule] = state.get("tests", [])
    retries = state.get("retries", 0) + 1

    # 把问题按测试名映射回模块
    name_to_mod: dict[str, TestModule] = {}
    for m in modules:
        for c in m.cases:
            name_to_mod[c.target] = m
    per_mod: dict[str, list] = {}
    for issue in check.issues:
        if issue.level != "error":
            continue
        mod = name_to_mod.get(issue.test or "")
        if mod is None and modules:
            mod = modules[0]
        if mod is not None:
            per_mod.setdefault(mod.path, []).append(issue)

    repaired: list[TestModule] = []
    for m in modules:
        issues = per_mod.get(m.path, [])
        if not issues:
            repaired.append(m)
            continue
        fixed = repair_module(state["llm"], m, issues, index, state.get("findings", []))
        repaired.append(fixed)

    ms = (time.time() - t0) * 1000
    _ev(state, "repair", "repaired",
        f"第 {retries} 轮修复：{sum(len(v) for v in per_mod.values())} 个错误", ms,
        mode="llm" if state["llm"].enabled else "drop-invalid")
    _tick(state, "repair", ms)
    return {"tests": repaired, "retries": retries}


# --------------------------------------------------------------------------- 9
def report_node(state: AgentState) -> dict:
    from ..schemas import Report

    t0 = time.time()
    req: AnalyzeRequest = state["request"]
    stats = state["stats"]
    check = state.get("check") or CheckReport()
    modules = state.get("tests", [])
    llm = state["llm"]

    timings = dict(state.get("timings", {}))
    timings["total"] = round(sum(timings.values()) + (time.time() - t0) * 1000, 1)

    report = Report(
        job_id=state["job_id"],
        repo=stats.repo,
        stats=stats,
        targets=state.get("targets", []),
        findings=state.get("findings", []),
        tests=modules,
        check=check,
        candidates_total=len(state.get("candidates", [])),
        llm_calls=llm.calls,
        timings_ms=timings,
        trace=state.get("trace", []),
        llm_enabled=llm.enabled,
        notes=state.get("notes", []) + ([llm.last_error] if llm.last_error and not llm.enabled else []),
    )

    out_dir = SETTINGS.storage_dir / state["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tests_dir = out_dir / "tests_generated"
    tests_dir.mkdir(exist_ok=True)
    for m in modules:
        target = tests_dir / Path(m.path).name
        target.write_text(m.code, encoding="utf-8")

    _ev(state, "report", "written", f"报告：{out_dir / 'report.json'}", (time.time() - t0) * 1000,
        findings=len(report.findings), tests=sum(len(m.cases) for m in modules))
    return {"timings": timings, "trace": state.get("trace", [])}
