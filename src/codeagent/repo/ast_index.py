"""AST 静态索引：把仓库解析成"符号 + 调用图 + 导入表"。

只做语法树分析，不 import 任何被分析代码。
"""
from __future__ import annotations

import ast
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..schemas import Chunk, ImportRef, ParseError, Symbol, SymbolKind

CONTROL_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith,
                 ast.Match if hasattr(ast, "Match") else ast.If, ast.IfExp)


def module_name(rel_path: str) -> str:
    p = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    parts = [x for x in p.split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Metrics(ast.NodeVisitor):
    """圈复杂度 + 最大嵌套深度（不进入嵌套函数体，嵌套函数各自统计）。"""

    def __init__(self) -> None:
        self.complexity = 1
        self.max_depth = 0
        self._depth = 0

    def _enter(self, node: ast.AST) -> None:
        self.complexity += 1
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)

    def _exit(self, node: ast.AST) -> None:
        self._depth -= 1

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not self._root:
            return
        super().generic_visit(node)

    def visit(self, node: ast.AST) -> None:  # type: ignore[override]
        for attr in ("_root",):
            if not hasattr(self, attr):
                self._root = node
        return super().visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_For(self, node: ast.For) -> None:
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_While(self, node: ast.While) -> None:
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += max(0, len(node.items) - 1)
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.complexity += max(0, len(node.items) - 1)
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.AST) -> None:
        cases = getattr(node, "cases", [])
        self.complexity += max(0, len(cases) - 1)
        self._enter(node)
        self.generic_visit(node)
        self._exit(node)


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    """返回 (参数总数, 必填参数数)，不计 self/cls。"""
    a = node.args
    positional = [x for x in (a.posonlyargs + a.args) if x.arg not in {"self", "cls"}]
    required = max(0, len(positional) - len(a.defaults))
    required += sum(1 for i, _ in enumerate(a.kwonlyargs) if a.kw_defaults[i] is None)
    total = len(positional) + len(a.kwonlyargs)
    return total, required


def _fill_doc_first_line(doc: str | None) -> str | None:
    if not doc:
        return None
    return doc.strip().splitlines()[0][:160]


class CodeIndex:
    """仓库级符号索引。"""

    def __init__(self) -> None:
        self.symbols: dict[str, Symbol] = {}
        self.by_file: dict[str, list[str]] = defaultdict(list)
        self.imports: dict[str, list[ImportRef]] = defaultdict(list)
        self.unused_imports: list[tuple[str, str, int]] = []   # (file, name, lineno)
        self.parse_errors: list[ParseError] = []
        self.files: list[str] = []
        self.loc: int = 0
        self.callers: dict[str, list[str]] = defaultdict(list)
        self.modules: dict[str, str] = {}                      # module -> file
        self._by_name: dict[str, list[str]] = defaultdict(list)

    # ---------- 构建 ----------
    def build(self, root: Path, rel_files: Iterable[str]) -> "CodeIndex":
        for rel in rel_files:
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                self.parse_errors.append(ParseError(file=rel, message=f"读取失败：{exc}"))
                continue
            self.files.append(rel)
            self.loc += text.count("\n") + 1
            self.modules[module_name(rel)] = rel
            try:
                tree = ast.parse(text, filename=rel, type_comments=True)
            except SyntaxError as exc:
                self.parse_errors.append(ParseError(file=rel, message=exc.msg, lineno=exc.lineno))
                continue
            self._collect(tree, rel, text)
        self._resolve_graph()
        return self

    def _collect(self, tree: ast.Module, rel: str, text: str) -> None:
        lines = text.splitlines()
        self._collect_imports(tree, rel)

        module_symbol = Symbol(
            id=f"{rel}::<module>",
            file=rel,
            qualname="<module>",
            name=Path(rel).stem,
            kind=SymbolKind.MODULE,
            lineno=1,
            end_lineno=max(1, len(lines)),
            docstring=_fill_doc_first_line(ast.get_docstring(tree)),
            source=text,
            callees=sorted(_call_names(tree)),
        )
        self.symbols[module_symbol.id] = module_symbol
        self.by_file[rel].append(module_symbol.id)

        def visit_body(body: list[ast.stmt], prefix: str, parent: str | None) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    self._add_class(node, rel, lines, prefix, parent, visit_body, text)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._add_function(node, rel, lines, prefix, parent, text)

        visit_body(tree.body, "", None)
        self._collect_unused_imports(tree, rel)

    def _add_class(self, node: ast.ClassDef, rel: str, lines: list[str], prefix: str,
                   parent: str | None, visit_body, text: str) -> None:
        qual = f"{prefix}{node.name}"
        sid = f"{rel}::{qual}"
        seg = ast.get_source_segment(text, node) or ""
        sym = Symbol(
            id=sid,
            file=rel,
            qualname=qual,
            name=node.name,
            kind=SymbolKind.CLASS,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            bases=[ast.unparse(b) for b in node.bases],
            docstring=_fill_doc_first_line(ast.get_docstring(node)),
            source=textwrap.dedent(seg),
            parent=parent,
        )
        self.symbols[sid] = sym
        self.by_file[rel].append(sid)
        self._by_name[node.name].append(sid)
        visit_body(node.body, f"{qual}.", sid)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, rel: str,
                      lines: list[str], prefix: str, parent: str | None, text: str) -> None:
        qual = f"{prefix}{node.name}"
        sid = f"{rel}::{qual}"
        seg = ast.get_source_segment(text, node) or ""
        metrics = _Metrics()
        metrics.visit(node)
        total, required = _signature(node)
        sym = Symbol(
            id=sid,
            file=rel,
            qualname=qual,
            name=node.name,
            kind=SymbolKind.METHOD if prefix else SymbolKind.FUNCTION,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            n_params=total,
            required_params=required,
            complexity=metrics.complexity,
            max_nesting=metrics.max_depth,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=[ast.unparse(d) for d in node.decorator_list],
            docstring=_fill_doc_first_line(ast.get_docstring(node)),
            source=textwrap.dedent(seg),
            parent=parent,
            callees=sorted(_call_names(node)),
        )
        self.symbols[sid] = sym
        self.by_file[rel].append(sid)
        self._by_name[node.name].append(sid)

    def _collect_imports(self, tree: ast.Module, rel: str) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[rel].append(
                        ImportRef(module=a.name, alias=a.asname or a.name.split(".")[0], lineno=node.lineno)
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.names and node.names[0].name == "*":
                    self.imports[rel].append(ImportRef(module=mod, name="*", lineno=node.lineno))
                    continue
                for a in node.names:
                    self.imports[rel].append(
                        ImportRef(module=mod, name=a.name, alias=a.asname or a.name, lineno=node.lineno)
                    )

    def _collect_unused_imports(self, tree: ast.Module, rel: str) -> None:
        imported: dict[str, tuple[int, str]] = {}
        for node in tree.body:  # 只看模块顶层导入
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported[a.asname or a.name.split(".")[0]] = (node.lineno, a.name)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name != "*":
                        imported[a.asname or a.name] = (node.lineno, a.name)

        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                root = node
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    used.add(root.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                used.add(node.value)

        # __all__ 里出现的名字不算未使用
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                for elt in getattr(node.value, "elts", []):
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        used.add(elt.value)

        for name, (lineno, orig) in imported.items():
            if name not in used and name != "annotations":
                self.unused_imports.append((rel, orig, lineno))

    def _resolve_graph(self) -> None:
        for name, ids in self._by_name.items():
            ids.sort(key=lambda s: (self.symbols[s].kind != SymbolKind.FUNCTION, len(s)))
        for sid, sym in self.symbols.items():
            for callee in sym.callees:
                target = self.resolve_name(callee, preferred_file=sym.file)
                if target and target != sid:
                    self.callers[target].append(sid)

    # ---------- 查询 ----------
    def resolve_name(self, name: str, preferred_file: str | None = None) -> str | None:
        ids = self._by_name.get(name)
        if not ids:
            return None
        if preferred_file:
            same = [i for i in ids if self.symbols[i].file == preferred_file]
            if same:
                return same[0]
        return ids[0]

    def get(self, symbol_id: str) -> Symbol | None:
        return self.symbols.get(symbol_id)

    def callers_of(self, symbol_id: str, limit: int = 8) -> list[Symbol]:
        return [self.symbols[s] for s in self.callers.get(symbol_id, [])[:limit] if s in self.symbols]

    def functions(self) -> list[Symbol]:
        return [s for s in self.symbols.values() if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)]

    def imports_of(self, file: str) -> list[ImportRef]:
        return self.imports.get(file, [])

    def module_for_file(self, file: str) -> str:
        for mod, f in self.modules.items():
            if f == file:
                return mod
        return module_name(file)

    # ---------- 检索单元 ----------
    def to_chunks(self, max_chars: int = 4000) -> list[Chunk]:
        """函数/类级切片。

        纯脚本文件（没有任何函数/类）才保留 module 块——否则整文件块会和每个函数块
        抢同一批词，把 BM25/向量的排名搅浑。
        """
        chunks: list[Chunk] = []
        files_with_symbols = {
            s.file for s in self.symbols.values() if s.kind != SymbolKind.MODULE
        }
        for sym in self.symbols.values():
            if sym.kind == SymbolKind.MODULE:
                if sym.file in files_with_symbols:
                    continue
                body = sym.source[:max_chars]
                header = f"# file: {sym.file}\n# module: {module_name(sym.file)}\n"
            else:
                head = f"# file: {sym.file}:{sym.lineno}\n# {sym.kind.value}: {sym.qualname}"
                if sym.decorators:
                    head += f"\n# decorators: {', '.join(sym.decorators)}"
                if sym.bases:
                    head += f"\n# bases: {', '.join(sym.bases)}"
                if sym.docstring:
                    head += f"\n# doc: {sym.docstring}"
                header = head + "\n"
                body = sym.source[:max_chars]
            text = header + body
            chunks.append(
                Chunk(
                    id=sym.id,
                    symbol_id=sym.id,
                    file=sym.file,
                    lineno=sym.lineno,
                    end_lineno=sym.end_lineno,
                    kind=sym.kind,
                    text=text,
                    tokens=len(text) // 4,
                )
            )
        return chunks

    # ---------- 统计 ----------
    def stats(self):  # -> RepoStats 片段
        funcs = [s for s in self.symbols.values() if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)]
        classes = [s for s in self.symbols.values() if s.kind == SymbolKind.CLASS]
        return {
            "files": len(self.files),
            "symbols": len(self.symbols),
            "functions": len(funcs),
            "classes": len(classes),
            "loc": self.loc,
            "parse_errors": self.parse_errors,
        }


def tokenize_code(text: str) -> list[str]:
    """代码友好的分词：标识符 + 驼峰/下划线拆分 + 中文双字。

    先按大小写切分再小写，否则 CamelCase 会被整体压平（OrderItem -> orderitem）。
    """
    toks: list[str] = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]{2}", text):
        if "\u4e00" <= w[0] <= "\u9fff":
            toks.append(w)
            continue
        for part in re.split(r"_+", w):
            if not part:
                continue
            subs = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+", part) or [part]
            toks.extend(s.lower() for s in subs)
    return [t for t in toks if len(t) > 1 or t.isdigit()]
