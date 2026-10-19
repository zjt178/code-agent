"""生成结果的静态自检。

约束：本项目不执行任何被分析代码，也不执行生成的测试，因此"自检"全部是语法树层面的校验：
1) 语法是否可解析；
2) import 的模块/符号是否真实存在（对照 AST 索引，而不是 import 试一下）；
3) 测试函数命名、断言、重复定义；
4) 是否引用了未定义名字；
5) 是否存在网络/文件/子进程等副作用调用（需要 mock 才能跑）；
6) 目标函数是否真的被判到（覆盖率）。
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict

from ..repo.ast_index import CodeIndex
from ..schemas import CheckIssue, CheckReport, TestCase, TestModule

STDLIB = set(sys.stdlib_module_names) | {"pytest", "typing_extensions", "__future__"}
FORBIDDEN_CALLS = {
    "open": "文件读写",
    "input": "标准输入",
    "eval": "动态执行",
    "exec": "动态执行",
    "compile": "动态执行",
    "os.system": "命令执行",
    "os.popen": "命令执行",
    "subprocess.run": "子进程",
    "subprocess.Popen": "子进程",
    "shutil.rmtree": "删除目录",
    "time.sleep": "真实等待",
    "socket.socket": "网络",
}


def _repo_top_levels(index: CodeIndex) -> set[str]:
    tops: set[str] = set()
    for mod in index.modules:
        tops.add(mod.split(".")[0])
    for f in index.files:
        if "/" not in f:
            tops.add(f[:-3])
    return tops


def _module_symbols(index: CodeIndex, module: str) -> set[str] | None:
    """返回某模块导出的顶层名字；模块不在仓库内则返回 None。"""
    file = index.modules.get(module)
    if file is None and module.endswith(".__init__"):
        file = index.modules.get(module[: -len(".__init__")])
    if file is None:
        return None
    names: set[str] = set()
    for sid in index.by_file.get(file, []):
        sym = index.symbols[sid]
        if sym.kind.value == "module":
            continue
        names.add(sym.qualname.split(".")[0])
    return names


def _call_name(func: ast.AST) -> str:
    try:
        return ast.unparse(func)
    except Exception:
        return ""


def _collect_asserts(body: list[ast.stmt]) -> int:
    n = 0
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assert):
            n += 1
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr.func).startswith(
                    "pytest.raises"
                ):
                    n += 1
    return n


def check_module(module: TestModule, index: CodeIndex) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    try:
        tree = ast.parse(module.code)
    except SyntaxError as exc:
        return [CheckIssue(level="error", code="SYNTAX", message=f"第 {exc.lineno} 行语法错误：{exc.msg}")]

    tops = _repo_top_levels(index)
    imported: set[str] = {"pytest"}

    # ---- import 校验 ----
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                imported.add(a.asname or root)
                if root not in STDLIB and root not in tops:
                    issues.append(
                        CheckIssue(level="error", code="UNKNOWN_IMPORT",
                                   message=f"导入的模块 {a.name} 既不在标准库也不在仓库内")
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            for a in node.names:
                if a.name == "*":
                    # 星号导入：把该模块真实导出的顶层名字全部视为已定义
                    avail = _module_symbols(index, mod)
                    if avail:
                        imported |= avail
                    else:
                        imported.add(a.name)
                else:
                    imported.add(a.asname or a.name)
            if node.level:  # 相对导入，视为仓库内
                continue
            if root not in STDLIB and root not in tops:
                issues.append(
                    CheckIssue(level="error", code="UNKNOWN_IMPORT",
                               message=f"from {mod} import ... 的模块不在仓库内")
                )
                continue
            if root in tops and node.names and node.names[0].name != "*":
                avail = _module_symbols(index, mod)
                if avail is not None:
                    missing = [a.name for a in node.names if a.name not in avail and a.name not in STDLIB]
                    if missing:
                        issues.append(
                            CheckIssue(level="error", code="UNKNOWN_SYMBOL",
                                       message=f"{mod} 中不存在符号：{', '.join(missing)}")
                        )

    # ---- 测试函数 ----
    tests: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            if node.name in tests:
                issues.append(CheckIssue(level="error", code="DUPLICATE_TEST",
                                         message=f"测试函数重名：{node.name}", test=node.name))
            tests[node.name] = node

    if not tests:
        issues.append(CheckIssue(level="error", code="NO_TESTS", message="文件中没有 test_ 开头的测试函数"))

    # ---- 未定义名字 ----
    import builtins as _b

    builtins = set(dir(_b))
    module_level = set(imported) | {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    module_level |= {
        t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
    }
    module_level |= {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}

    for name, fn in tests.items():
        local: set[str] = set(module_level)
        local |= {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
        local |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                local |= {a.asname or a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                local |= {a.asname or a.name for a in node.names}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local and n.id not in builtins:
                issues.append(
                    CheckIssue(level="error", code="UNRESOLVED_NAME",
                               message=f"引用了未定义的名字 {n.id}", test=name)
                )
                break

    # ---- 断言 / 副作用 ----
    for name, fn in tests.items():
        if _collect_asserts(fn.body) == 0:
            issues.append(CheckIssue(level="warn", code="NO_ASSERT", message="测试没有任何断言", test=name))
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                cname = _call_name(n.func)
                base = cname.split("(")[0]
                root = ".".join(base.split(".")[:2])
                hit = FORBIDDEN_CALLS.get(base) or FORBIDDEN_CALLS.get(root)
                if hit:
                    issues.append(
                        CheckIssue(level="warn", code="SIDE_EFFECT",
                                   message=f"{base}() 有{hit}副作用，运行前需 mock 或改用 tmp_path",
                                   test=name)
                    )

    # ---- 用例状态回填 ----
    by_test: dict[str, list[CheckIssue]] = defaultdict(list)
    for issue in issues:
        if issue.test:
            by_test[issue.test].append(issue)
    for case in module.cases:
        own = by_test.get(case.target, [])
        case.issues = [f"[{i.code}] {i.message}" for i in own]
        if any(i.level == "error" for i in own):
            case.status = "rejected"
        elif own:
            case.status = "warn"
        else:
            case.status = "valid"
    return issues


def check_all(modules: list[TestModule], index: CodeIndex, targets: list[str]) -> CheckReport:
    all_issues: list[CheckIssue] = []
    valid = 0
    total_cases = 0
    covered: set[str] = set()
    for mod in modules:
        all_issues.extend(check_module(mod, index))
        for case in mod.cases:
            total_cases += 1
            if case.status != "rejected":
                valid += 1
                covered.add(case.target_symbol_id)

    untested = [t for t in targets if t not in covered]
    errors = [i for i in all_issues if i.level == "error"]
    return CheckReport(
        ok=not errors,
        issues=all_issues,
        checked_tests=total_cases,
        valid_tests=valid,
        coverage_ratio=(len(targets) - len(untested)) / len(targets) if targets else 0.0,
        untested_targets=untested,
    )
