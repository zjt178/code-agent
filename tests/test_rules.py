"""规则引擎单测：每条规则至少一个正例，另测一个干净函数不误报。"""
from __future__ import annotations

from pathlib import Path

import pytest

from codeagent.analysis.rules import run_rules
from codeagent.repo.ast_index import CodeIndex

BUGGY = '''
import os
import pickle

def f(a, b=[]):
    try:
        x = 1
    except:
        pass
    if a == None:
        return 0
    if b is "x":
        return 1
    return 2
    print("dead")

def g(sql):
    cur = None
    cur.execute(f"select * from t where id = {sql}")

def h(p, q, r, s, t, u):
    token = "sk-live-abcdefgh"
    return eval(p)

def k():
    os.system("ls")
'''

CLEAN = '''
"""模块文档。"""
from pathlib import Path


def read(path: str) -> str:
    """读取文件内容。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        return fh.read()


def total(items: list[int]) -> int:
    """求和。"""
    return sum(items)
'''


def _index(tmp_path: Path, source: str) -> CodeIndex:
    (tmp_path / "m.py").write_text(source, encoding="utf-8")
    return CodeIndex().build(tmp_path, ["m.py"])


def test_rules_fire(tmp_path: Path) -> None:
    idx = _index(tmp_path, BUGGY)
    ids = {c.rule_id for c in run_rules(idx)}
    expected = {"bare-except", "except-pass", "mutable-default-arg", "eq-none", "is-literal",
                "unreachable-code", "sql-fstring", "hardcoded-secret", "eval-exec", "os-system",
                "too-many-params"}
    assert expected <= ids, expected - ids


def test_clean_code_has_no_findings(tmp_path: Path) -> None:
    idx = _index(tmp_path, CLEAN)
    assert run_rules(idx) == []


def test_snippet_has_absolute_lineno(tmp_path: Path) -> None:
    idx = _index(tmp_path, BUGGY)
    cands = [c for c in run_rules(idx) if c.rule_id == "bare-except"]
    assert cands and cands[0].lineno == 8
    assert "except" in cands[0].snippet


@pytest.mark.parametrize("rule", ["sql-fstring", "hardcoded-secret"])
def test_security_rules_are_critical(tmp_path: Path, rule: str) -> None:
    idx = _index(tmp_path, BUGGY)
    hit = [c for c in run_rules(idx) if c.rule_id == rule]
    assert hit and hit[0].severity.value == "critical"
