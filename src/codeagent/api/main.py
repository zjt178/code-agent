"""FastAPI 服务：提交分析任务 → 轮询状态 → 取报告。

任务放后台线程跑（分析是 CPU/IO 混合，单机场景足够；生产应换队列）。
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..agent.graph import run_analysis
from ..schemas import AnalyzeRequest, Report

app = FastAPI(title="CodeAgent API", version="0.1.0",
              description="代码仓库静态分析 Agent：AST 解析 + 混合检索 + 缺陷识别 + 测试生成 + 静态自检（不执行代码）")

JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(job_id: str, req: AnalyzeRequest) -> None:
    try:
        report = run_analysis(req, job_id)
        with _LOCK:
            JOBS[job_id].update(
                status="done",
                finished_at=_now(),
                summary={
                    "files": report.stats.files,
                    "findings": len(report.findings),
                    "tests": sum(len(m.cases) for m in report.tests),
                    "check_ok": report.check.ok,
                    "coverage": round(report.check.coverage_ratio, 3),
                    "llm_calls": report.llm_calls,
                },
                report=report,
            )
    except Exception as exc:  # noqa: BLE001 - 任务边界，必须兜住
        with _LOCK:
            JOBS[job_id].update(status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "jobs": len(JOBS)}


@app.post("/analyze", response_model=JobCreated)
def analyze(req: AnalyzeRequest) -> JobCreated:
    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        JOBS[job_id] = {"status": "running", "created_at": _now(), "error": None, "report": None}
    threading.Thread(target=_run, args=(job_id, req), daemon=True).start()
    return JobCreated(job_id=job_id, status="running")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return JobStatus(
        job_id=job_id, status=job["status"], created_at=job["created_at"],
        finished_at=job.get("finished_at"), error=job.get("error"), summary=job.get("summary"),
    )


@app.get("/jobs/{job_id}/report", response_model=Report)
def job_report(job_id: str) -> Report:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] != "done":
        raise HTTPException(409, f"任务未完成：{job['status']}")
    return job["report"]
