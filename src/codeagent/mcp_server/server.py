"""MCP 工具服务：把分析能力暴露成标准工具，供 Claude/其他 Agent 客户端调用。

启动（stdio）：
    PYTHONPATH=src python -m codeagent.mcp_server.server

设计：索引按仓库路径缓存，避免每次工具调用都重建。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..agent.graph import run_analysis
from ..repo.ast_index import CodeIndex
from ..repo.loader import load_repo
from ..retrieval.embed import build_embedder
from ..retrieval.hybrid import HybridRetriever
from ..schemas import AnalyzeRequest

mcp = FastMCP("codeagent")

_CACHE: dict[str, tuple[CodeIndex, HybridRetriever]] = {}


def _index_for(repo: str) -> tuple[CodeIndex, HybridRetriever]:
    key = str(Path(repo).resolve())
    if key not in _CACHE:
        loaded = load_repo(repo)
        index = CodeIndex().build(loaded.root, loaded.rel_files)
        embedder, note = build_embedder()
        _CACHE[key] = (index, HybridRetriever(index.to_chunks(), embedder, note))
    return _CACHE[key]


@mcp.tool()
def analyze_repo(repo: str, max_targets: int = 6, use_llm: bool = False) -> str:
    """分析一个 Python 仓库（Git URL 或本地路径），返回缺陷、生成测试与自检结果摘要。

    repo: 仓库地址或本地路径；max_targets: 深度审查/生成测试的函数数；
    use_llm: 是否调用本地模型做复核（慢，但能过滤误报）。
    """
    report = run_analysis(AnalyzeRequest(repo=repo, max_targets=max_targets, use_llm=use_llm),
                          job_id=f"mcp-{uuid.uuid4().hex[:6]}")
    return json.dumps({
        "repo": report.repo,
        "stats": report.stats.model_dump(mode="json"),
        "findings": [f.model_dump(mode="json") for f in report.findings[:20]],
        "tests": sum(len(m.cases) for m in report.tests),
        "check": report.check.model_dump(mode="json"),
        "timings_ms": report.timings_ms,
    }, ensure_ascii=False)


@mcp.tool()
def search_code(repo: str, query: str, k: int = 6) -> str:
    """在仓库中做混合检索（BM25 + 向量 + RRF），返回最相关的函数片段。"""
    _, retriever = _index_for(repo)
    return json.dumps([h.as_dict() for h in retriever.search(query, k=k)], ensure_ascii=False)


@mcp.tool()
def list_symbols(repo: str, file: str | None = None) -> str:
    """列出仓库（或指定文件）的函数/类清单及其复杂度。"""
    index, _ = _index_for(repo)
    items = []
    for sid, sym in index.symbols.items():
        if sym.kind.value == "module":
            continue
        if file and sym.file != file:
            continue
        items.append({"symbol_id": sid, "kind": sym.kind.value, "lineno": sym.lineno,
                      "complexity": sym.complexity, "lines": sym.n_lines})
    return json.dumps(items[:400], ensure_ascii=False)


@mcp.tool()
def get_symbol(repo: str, symbol_id: str) -> str:
    """按 symbol_id（file::qualname）取源码与调用关系。"""
    index, _ = _index_for(repo)
    sym = index.get(symbol_id)
    if not sym:
        return json.dumps({"error": "not found"}, ensure_ascii=False)
    return json.dumps({
        "symbol_id": sym.id, "source": sym.source,
        "callers": [c.id for c in index.callers_of(sym.id)],
        "complexity": sym.complexity, "docstring": sym.docstring,
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
