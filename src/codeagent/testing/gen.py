"""单元测试生成。

两条通路：
- LLM 通路：给模型函数源码 + 同文件上下文 + 已确认缺陷，要求产出 pytest 文件；
- 模板通路：无 LLM 时生成"契约测试"（签名一致性 + 必填参数抛 TypeError），
  这类测试不触发业务副作用，且 100% 可静态验证。

无论哪条通路，产物都只是文本，不执行。
"""
from __future__ import annotations

import ast
import re
import textwrap

from ..repo.ast_index import CodeIndex
from ..schemas import CheckIssue, Finding, Symbol, SymbolKind, TestCase, TestModule
from ..analysis.rules import RULES

SYSTEM_PROMPT = """你是资深 Python 测试工程师。你要为给定的函数生成 pytest 单元测试。

硬性要求：
1. 只输出一个 Python 代码块，不要解释；
2. 只允许 import：标准库、pytest、以及被测仓库内的真实模块（必须用给定导入路径）；
3. 禁止网络、文件读写、子进程、sleep、随机数；被测函数若依赖外部资源，不要测它；
4. 每个测试函数必须包含至少一个具体断言（期望值要写清楚，不要只 assert result）；
5. 测试函数命名格式：test_<被测函数名>_<场景>；
6. 覆盖：正常路径、边界值、异常输入；不要为同一个函数重复写等价用例；
7. 不修改被测代码，不 mock 未导入的对象；
8. 所有用到的名字必须在文件顶部显式 import（禁止使用 requests_mock / responses / freezegun 等未声明的三方库）；
9. 需要 mock 时优先用 pytest 的 monkeypatch 或 unittest.mock（并在顶部 import）；
10. 测试文件必须自包含：用到的常量、fixture、helper 都要在本文件定义。"""


def _module_path(index: CodeIndex, file: str) -> str:
    return index.module_for_file(file)


def _parse_args(sym: Symbol) -> tuple[list[str], int]:
    try:
        tree = ast.parse(textwrap.dedent(sym.source))
    except SyntaxError:
        return [], 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.posonlyargs + node.args.args if a.arg not in {"self", "cls"}]
            args += [a.arg for a in node.args.kwonlyargs]
            required = len(args) - len(node.args.defaults) - sum(
                1 for d in node.args.kw_defaults if d is not None
            )
            return args, max(0, required)
    return [], 0


def _header(module_path: str, file: str) -> str:
    return (
        f'"""针对 {file} 自动生成的单元测试（静态分析产物，未经执行）。"""\n'
        f"import inspect\n\n"
        f"import pytest\n\n"
        f"from {module_path} import *  # noqa: F401,F403\n"
    )


def _template_case(sym: Symbol, module_path: str) -> TestCase | None:
    if sym.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
        return None
    args, required = _parse_args(sym)
    is_method = sym.kind == SymbolKind.METHOD
    indent = "    "

    if is_method:
        cls = (sym.qualname.split(".")[0]) if "." in sym.qualname else None
        if not cls:
            return None
        body = (
            f"def test_{sym.name}_is_callable():\n"
            f"{indent}assert hasattr({cls}, {sym.name!r})\n"
            f"{indent}assert callable(getattr({cls}, {sym.name!r}))\n"
        )
        name = f"test_{sym.name}_is_callable"
    else:
        body = (
            f"def test_{sym.name}_signature():\n"
            f"{indent}sig = inspect.signature({sym.name})\n"
            f"{indent}assert list(sig.parameters) == {args!r}\n"
        )
        name = f"test_{sym.name}_signature"
        if required > 0:
            body += (
                f"\n\ndef test_{sym.name}_missing_args_raises_type_error():\n"
                f"{indent}with pytest.raises(TypeError):\n"
                f"{indent * 2}{sym.name}()\n"
            )
    return TestCase(
        id=f"{sym.id}::{name}",
        target_symbol_id=sym.id,
        target=name,
        file=sym.file,
        code=body,
        focus="edge",
        generator="template",
    )


def _split_code(code: str) -> tuple[str, list[tuple[str, str]]]:
    """取模型返回的完整文件 + 其中的测试函数列表。

    注意：绝不能只保留 import 行再拼测试函数——模型常把常量、fixture、helper
    放在顶层，剥掉就会产生"未定义名字"。这里完整保留文件，只额外抽取用例元信息。
    """
    code = code.strip()
    fence = re.search(r"```(?:python)?\s*(.+?)```", code, re.S)
    if fence:
        code = fence.group(1).strip() + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []
    lines = code.splitlines()
    tests = [
        (n.name, "\n".join(lines[n.lineno - 1: n.end_lineno]))
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]
    return code, tests


def _drop_tests(code: str, names: set[str]) -> str:
    """按名称整段删除测试函数（用于无 LLM 时的降级修复）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    spans = [
        (n.lineno, n.end_lineno) for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    if not spans:
        return code
    lines = code.splitlines()
    drop: set[int] = set()
    for start, end in spans:
        drop.update(range(start, end + 1))
    kept = [ln for i, ln in enumerate(lines, 1) if i not in drop]
    return "\n".join(kept).rstrip() + "\n"


def _match_target(name: str, targets: list[Symbol]) -> Symbol | None:
    best: Symbol | None = None
    for sym in targets:
        if name.startswith(f"test_{sym.name}_") or name == f"test_{sym.name}":
            if best is None or len(sym.name) > len(best.name):
                best = sym
    return best


def _llm_file(llm, index: CodeIndex, file: str, targets: list[Symbol],
               findings: list[Finding], feedback: list[CheckIssue] | None = None) -> str | None:
    module = _module_path(index, file)
    blocks = []
    for sym in targets[:6]:
        rules = [f for f in findings if f.symbol_id == sym.id]
        rule_txt = "；".join(f"{f.rule_id}@{f.lineno}" for f in rules) or "无"
        blocks.append(
            f"### {sym.qualname}（{sym.kind.value}，第 {sym.lineno}-{sym.end_lineno} 行）\n"
            f"已确认缺陷：{rule_txt}\n```python\n{sym.source[:1800]}\n```"
        )
    imports = ", ".join(sorted({i.module for i in index.imports_of(file)})[:12]) or "无"
    fb = ""
    if feedback:
        fb = "\n上一版存在以下静态检查问题，请修复：\n" + "\n".join(
            f"- [{i.code}] {i.message}" for i in feedback[:12]
        )
    user = (
        f"仓库内模块路径：{module}\n源文件：{file}\n该文件已有的 import：{imports}\n\n"
        + "\n\n".join(blocks)
        + f"\n\n请只针对以上函数生成 pytest 测试文件（完整可运行，顶部包含必要 import）。{fb}"
    )
    return llm.chat(SYSTEM_PROMPT, user)


def generate_for_file(llm, index: CodeIndex, file: str, target_ids: list[str],
                      findings: list[Finding], feedback: list[CheckIssue] | None = None) -> TestModule:
    targets = [index.symbols[t] for t in target_ids if t in index.symbols]
    module_path = _module_path(index, file)
    path = f"tests_generated/test_{file.replace('/', '_')[:-3]}.py"

    if llm is not None and getattr(llm, "enabled", False) and targets:
        code = _llm_file(llm, index, file, targets, findings, feedback)
        if code:
            full_code, raw_tests = _split_code(code)
            cases: list[TestCase] = []
            for name, src in raw_tests:
                sym = _match_target(name, targets)
                if sym is None:
                    continue
                cases.append(
                    TestCase(
                        id=f"{sym.id}::{name}",
                        target_symbol_id=sym.id,
                        target=name,
                        file=file,
                        code=src,
                        focus="happy",
                        generator="llm",
                    )
                )
            if cases:
                return TestModule(path=path, code=full_code, cases=cases)

    # 模板兜底
    cases = [c for c in (_template_case(s, module_path) for s in targets) if c]
    header = (
        f'"""针对 {file} 的契约测试（无 LLM 模式自动生成，未经执行）。"""\n'
        f"import inspect\n\nimport pytest\n\n"
        f"from {module_path} import *  # noqa: F401,F403\n"
    )
    body = "\n\n".join(c.code for c in cases)
    return TestModule(path=path, code=header + "\n" + body + "\n", cases=cases)


def generate_tests(llm, index: CodeIndex, targets: list[str], findings: list[Finding],
                   feedback: dict[str, list[CheckIssue]] | None = None,
                   per_file_limit: int = 4) -> list[TestModule]:
    """按文件分组生成，避免每个函数一次 LLM 调用（成本可控）。"""
    by_file: dict[str, list[str]] = {}
    for sid in targets:
        sym = index.symbols.get(sid)
        if sym is None or sym.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            continue
        by_file.setdefault(sym.file, []).append(sid)

    modules: list[TestModule] = []
    for file, ids in by_file.items():
        fb = (feedback or {}).get(file)
        modules.append(generate_for_file(llm, index, file, ids[:per_file_limit], findings, fb))
    return modules


def repair_module(llm, module: TestModule, issues: list[CheckIssue], index: CodeIndex,
                  findings: list[Finding]) -> TestModule:
    """自检失败后的修复：LLM 重写；无 LLM 时剔除不合法用例并重建文件。"""
    file = module.cases[0].file if module.cases else ""
    target_ids = sorted({c.target_symbol_id for c in module.cases})
    if llm is not None and getattr(llm, "enabled", False) and file:
        fixed = generate_for_file(llm, index, file, target_ids, findings, feedback=issues)
        if fixed.cases:
            return fixed
    good = [c for c in module.cases if c.status != "rejected"]
    if not good:
        return module
    bad = {c.target for c in module.cases if c.status == "rejected"}
    code = _drop_tests(module.code, bad)
    return TestModule(path=module.path, code=code, cases=good)
