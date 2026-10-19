"""Agent 可调用的工具集（LangChain tool + MCP 共用同一实现）。

工具的职责边界：只读、只查索引，不执行被分析代码。
"""
from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from ..analysis.rules import run_rules
from ..repo.ast_index import CodeIndex
from ..retrieval.hybrid import HybridRetriever


def build_tools(index: CodeIndex, retriever: HybridRetriever) -> list[BaseTool]:
    @tool
    def search_code(query: str, k: int = 6) -> str:
        """语义+词法混合检索仓库代码。query 用自然语言或标识符描述要找的实现，
        例如 "解析 git url 并校验" 或 "retry backoff"；返回 file:line 与片段。"""
        hits = retriever.search(query, k=k)
        return json.dumps([h.as_dict() for h in hits], ensure_ascii=False)

    @tool
    def get_symbol(symbol_id: str) -> str:
        """按 symbol_id（形如 src/foo.py::Bar.baz）取完整源码、签名与指标。"""
        sym = index.get(symbol_id)
        if not sym:
            return json.dumps({"error": f"符号不存在：{symbol_id}"}, ensure_ascii=False)
        return json.dumps(
            {
                "symbol_id": sym.id,
                "file": sym.file,
                "lineno": sym.lineno,
                "end_lineno": sym.end_lineno,
                "kind": sym.kind.value,
                "complexity": sym.complexity,
                "max_nesting": sym.max_nesting,
                "n_params": sym.n_params,
                "docstring": sym.docstring,
                "source": sym.source,
            },
            ensure_ascii=False,
        )

    @tool
    def get_callers(symbol_id: str, limit: int = 5) -> str:
        """查某函数的调用方（谁依赖它），用于判断缺陷影响面。"""
        callers = index.callers_of(symbol_id, limit=limit)
        return json.dumps(
            [{"symbol_id": s.id, "file": s.file, "lineno": s.lineno, "source": s.source[:600]} for s in callers],
            ensure_ascii=False,
        )

    @tool
    def rule_scan(symbol_id: str) -> str:
        """对单个符号跑 AST 规则扫描，返回候选缺陷（未复核）。"""
        cands = run_rules(index, [symbol_id])
        return json.dumps([c.model_dump(mode="json") for c in cands], ensure_ascii=False)

    @tool
    def file_overview(file: str) -> str:
        """查看某个文件的符号清单与导入，用于了解代码结构。"""
        ids = index.by_file.get(file, [])
        syms = [
            {"symbol_id": i, "kind": index.symbols[i].kind.value, "lineno": index.symbols[i].lineno,
             "complexity": index.symbols[i].complexity}
            for i in ids
        ]
        imports = [f"{i.module}{'.' + i.name if i.name else ''}" for i in index.imports_of(file)]
        return json.dumps({"file": file, "symbols": syms, "imports": imports}, ensure_ascii=False)

    return [search_code, get_symbol, get_callers, rule_scan, file_overview]
