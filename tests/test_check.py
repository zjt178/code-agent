"""自检器单测：合法测试通过；错误 import / 未定义名 / 无断言 被识别。"""
from __future__ import annotations

from pathlib import Path

from codeagent.repo.ast_index import CodeIndex
from codeagent.schemas import TestCase, TestModule
from codeagent.testing.check import check_all, check_module

SRC = '''
"""模块。"""

def add(a, b):
    """相加。"""
    return a + b
'''


def _index(tmp_path: Path) -> CodeIndex:
    (tmp_path / "calc.py").write_text(SRC, encoding="utf-8")
    return CodeIndex().build(tmp_path, ["calc.py"])


def _module(code: str, target: str = "calc.py::add") -> TestModule:
    return TestModule(
        path="tests_generated/test_calc.py",
        code=code,
        cases=[TestCase(id="t1", target_symbol_id=target, target="test_add_ok", file="calc.py",
                        code="def test_add_ok():\n    assert True\n")],
    )


def test_valid_module_passes(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = (
        "from calc import add\n\n"
        "def test_add_ok():\n"
        "    assert add(1, 2) == 3\n"
    )
    issues = check_module(_module(code), idx)
    assert [i for i in issues if i.level == "error"] == []


def test_syntax_error(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    issues = check_module(_module("def test_x(:\n"), idx)
    assert issues[0].code == "SYNTAX"


def test_unknown_module(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = "import not_a_real_module\n\ndef test_add_ok():\n    assert True\n"
    codes = {i.code for i in check_module(_module(code), idx)}
    assert "UNKNOWN_IMPORT" in codes


def test_unknown_symbol(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = "from calc import missing\n\ndef test_add_ok():\n    assert True\n"
    codes = {i.code for i in check_module(_module(code), idx)}
    assert "UNKNOWN_SYMBOL" in codes


def test_unresolved_name(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = "def test_add_ok():\n    assert undefined_name == 1\n"
    codes = {i.code for i in check_module(_module(code), idx)}
    assert "UNRESOLVED_NAME" in codes


def test_no_assert_is_warning(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = "def test_add_ok():\n    x = 1\n"
    issues = check_module(_module(code), idx)
    assert any(i.code == "NO_ASSERT" and i.level == "warn" for i in issues)


def test_side_effect_is_warning(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = "def test_add_ok():\n    open('x', 'w')\n    assert True\n"
    issues = check_module(_module(code), idx)
    assert any(i.code == "SIDE_EFFECT" for i in issues)


def test_coverage_uses_planned_targets(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    code = "from calc import add\n\ndef test_add_ok():\n    assert add(1, 2) == 3\n"
    rep = check_all([_module(code)], idx, ["calc.py::add"])
    assert rep.coverage_ratio == 1.0
    rep2 = check_all([_module(code)], idx, ["calc.py::add", "calc.py::missing"])
    assert rep2.coverage_ratio == 0.5
