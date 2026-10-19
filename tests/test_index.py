"""AST 索引单测：符号、复杂度、嵌套、调用图、未使用导入。"""
from __future__ import annotations

from pathlib import Path

from codeagent.repo.ast_index import CodeIndex, module_name, tokenize_code

SRC = '''
import os
import json


class Repo:
    """示例类。"""

    def load(self, path):
        return json.loads(open(path).read())

    def save(self, path, data):
        with open(path, "w") as fh:
            json.dump(data, fh)


def helper(x):
    if x:
        for i in range(3):
            while i:
                i -= 1
    return x


def caller(y):
    return helper(y)
'''


def _build(tmp_path: Path) -> CodeIndex:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(SRC, encoding="utf-8")
    return CodeIndex().build(tmp_path, ["pkg/__init__.py", "pkg/mod.py"])


def test_module_name() -> None:
    assert module_name("pkg/mod.py") == "pkg.mod"
    assert module_name("pkg/__init__.py") == "pkg"


def test_symbols_and_metrics(tmp_path: Path) -> None:
    idx = _build(tmp_path)
    helper = idx.get("pkg/mod.py::helper")
    assert helper is not None
    assert helper.complexity >= 4          # if + for + while + base
    assert helper.max_nesting >= 3
    assert idx.get("pkg/mod.py::Repo.load").kind.value == "method"
    assert idx.get("pkg/mod.py::Repo").kind.value == "class"


def test_call_graph(tmp_path: Path) -> None:
    idx = _build(tmp_path)
    callers = [s.qualname for s in idx.callers_of("pkg/mod.py::helper")]
    assert "caller" in callers


def test_unused_import_detected(tmp_path: Path) -> None:
    idx = _build(tmp_path)
    unused = {name for _file, name, _line in idx.unused_imports}
    assert "os" in unused
    assert "json" not in unused


def test_tokenize_splits_snake_and_camel() -> None:
    toks = tokenize_code("def total_price(OrderItem): pass")
    assert "total" in toks and "price" in toks and "order" in toks


def test_chunks_cover_all_symbols(tmp_path: Path) -> None:
    idx = _build(tmp_path)
    chunks = idx.to_chunks()
    funcs = [s for s in idx.symbols.values() if s.kind.value != "module"]
    assert len([c for c in chunks if c.kind.value != "module"]) == len(funcs)
    assert all("file:" in c.text for c in chunks)
