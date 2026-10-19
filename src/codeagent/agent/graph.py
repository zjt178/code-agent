"""LangGraph 编排 + 统一入口 run_analysis()。"""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from ..analysis.llm import LLMClient
from ..config import SETTINGS
from ..schemas import AnalyzeRequest, CheckReport, Report
from .nodes import (
    AgentState, build_index_node, detect_node, gen_tests_node, investigate_node, load_repo_node,
    plan_node, repair_node, report_node, self_check_node,
)



def _after_plan(state: AgentState) -> str:
    return "investigate" if state.get("targets") else "report"


def _after_check(state: AgentState) -> str:
    check: CheckReport = state.get("check") or CheckReport()
    errors = [i for i in check.issues if i.level == "error"]
    if not errors:
        return "report"
    if state.get("retries", 0) < SETTINGS.max_repair:
        return "repair"
    return "report"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_repo", load_repo_node)
    g.add_node("build_index", build_index_node)
    g.add_node("plan", plan_node)
    g.add_node("investigate", investigate_node)
    g.add_node("detect", detect_node)
    g.add_node("gen_tests", gen_tests_node)
    g.add_node("self_check", self_check_node)
    g.add_node("repair", repair_node)
    g.add_node("report", report_node)

    g.set_entry_point("load_repo")
    g.add_edge("load_repo", "build_index")
    g.add_edge("build_index", "plan")
    g.add_conditional_edges("plan", _after_plan, {"investigate": "investigate", "report": "report"})
    g.add_edge("investigate", "detect")
    g.add_edge("detect", "gen_tests")
    g.add_edge("gen_tests", "self_check")
    g.add_conditional_edges(
        "self_check", _after_check, {"repair": "repair", "report": "report"}
    )
    g.add_edge("repair", "self_check")
    g.add_edge("report", END)
    return g.compile()


GRAPH = None


def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    return GRAPH


def run_analysis(req: AnalyzeRequest, job_id: str, llm_enabled: bool | None = None) -> Report:
    """跑完整条图，返回 Report（report 节点同时会把 json/测试文件落盘）。"""
    enabled = req.use_llm if llm_enabled is None else llm_enabled
    llm = LLMClient(enabled=enabled)
    state: dict[str, Any] = {
        "request": req,
        "job_id": job_id,
        "llm": llm,
        "trace": [],
        "timings": {},
        "notes": [],
        "retries": 0,
        "targets": [],
        "findings": [],
        "tests": [],
        "candidates": [],
        "evidence": {},
    }
    t0 = time.time()
    final = get_graph().invoke(state, config={"recursion_limit": 40})
    report = Report(
        job_id=job_id,
        repo=final["stats"].repo,
        stats=final["stats"],
        targets=final.get("targets", []),
        findings=final.get("findings", []),
        tests=final.get("tests", []),
        check=final.get("check") or CheckReport(),
        candidates_total=len(final.get("candidates", [])),
        llm_calls=llm.calls,
        timings_ms={**final.get("timings", {}), "wall": round((time.time() - t0) * 1000, 1)},
        trace=final.get("trace", []),
        llm_enabled=llm.enabled,
        notes=final.get("notes", []),
    )
    return report
