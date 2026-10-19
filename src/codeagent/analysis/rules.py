"""AST 规则引擎：静态、确定、可解释。

定位：负责"高召回"——宁可多报，交给 LLM 复核"高精度"。
每条规则都有 id / 类别 / 默认严重度 / 修复建议，报告里可直接引用。

不执行任何被分析代码：全部基于 ast 节点，不调用 eval/exec/import。
"""
from __future__ import annotations

import ast
import builtins
import textwrap
from dataclasses import dataclass
from typing import Iterable, Iterator

from ..repo.ast_index import CodeIndex
from ..schemas import Candidate, Severity, Symbol, SymbolKind


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    severity: Severity
    title: str
    suggestion: str


RULES: dict[str, Rule] = {}


def _rule(id: str, category: str, severity: Severity, title: str, suggestion: str) -> Rule:
    r = Rule(id, category, severity, title, suggestion)
    RULES[id] = r
    return r


# ---------------- 正确性 ----------------
_rule("bare-except", "correctness", Severity.HIGH, "裸 except 吞掉所有异常（含 KeyboardInterrupt）",
      "改为捕获具体异常类型，至少用 except Exception 并记录日志。")
_rule("except-pass", "correctness", Severity.HIGH, "except 分支只有 pass，错误被静默丢弃",
      "在 except 中记录日志、补偿或重新抛出，不要静默通过。")
_rule("mutable-default-arg", "correctness", Severity.HIGH, "可变对象作默认参数，跨调用共享状态",
      "默认值改为 None，在函数体内 if x is None: x = []。")
_rule("eq-none", "correctness", Severity.LOW, "用 == / != 与 None 比较",
      "改用 is None / is not None，避免重载 __eq__ 带来的语义偏差。")
_rule("is-literal", "correctness", Severity.MEDIUM, "用 is 比较字面量（身份比较语义错误）",
      "数值/字符串比较用 ==。")
_rule("unreachable-code", "correctness", Severity.MEDIUM, "return/raise/break 之后存在不可达代码",
      "删除不可达分支，或检查 return 位置是否写错。")
_rule("duplicate-dict-key", "correctness", Severity.MEDIUM, "字典字面量存在重复键，后者覆盖前者",
      "删除重复键，或改为运行时构建。")
_rule("shadow-builtin", "correctness", Severity.LOW, "局部变量/参数遮蔽内置名",
      "重命名变量（如 list -> items），避免后续调用内置函数失败。")
_rule("inconsistent-return", "correctness", Severity.MEDIUM, "函数同时存在 return 值 与 裸 return",
      "统一返回值：要么全部返回值，要么显式 return None 并在文档说明。")

# ---------------- 可靠性 ----------------
_rule("broad-except", "reliability", Severity.MEDIUM, "捕获过宽（Exception/BaseException）且未重新抛出",
      "收窄异常类型；确实需要兜底时记录日志并明确返回值。")
_rule("subprocess-shell-true", "reliability", Severity.HIGH, "subprocess 使用 shell=True，存在命令注入风险",
      "传列表参数并保持 shell=False。")
_rule("os-system", "reliability", Severity.HIGH, "使用 os.system/os.popen 执行命令",
      "改用 subprocess.run([...], check=True)。")
_rule("eval-exec", "reliability", Severity.HIGH, "使用 eval/exec/compile 执行动态代码",
      "避免动态执行；必须使用时限制输入来源并做白名单校验。")
_rule("requests-no-timeout", "reliability", Severity.MEDIUM, "HTTP 调用未设置 timeout，可能永久挂起",
      "显式传 timeout=（连接, 读取）。")
_rule("yaml-unsafe-load", "reliability", Severity.HIGH, "yaml.load 未指定 SafeLoader，可执行任意对象构造",
      "改用 yaml.safe_load 或 Loader=yaml.SafeLoader。")
_rule("pickle-load", "reliability", Severity.HIGH, "pickle.load/loads 反序列化不可信数据可导致 RCE",
      "改用 json 等安全格式，或校验数据来源与签名。")

# ---------------- 安全 ----------------
_rule("sql-fstring", "security", Severity.CRITICAL, "SQL 语句由字符串拼接/插值构造，存在注入风险",
      "使用参数化查询：cursor.execute(sql, params)。")
_rule("hardcoded-secret", "security", Severity.CRITICAL, "疑似硬编码密钥/口令",
      "改为从环境变量或密钥管理服务读取，并轮换已泄露的凭据。")

# ---------------- 资源 ----------------
_rule("open-without-with", "resources", Severity.MEDIUM, "文件句柄未使用 with，异常时不会关闭",
      "改用 with open(...) as f: 或 try/finally 关闭。")

# ---------------- 可维护性 ----------------
_rule("high-complexity", "maintainability", Severity.MEDIUM, "圈复杂度超过阈值（>10）",
      "拆分函数、提前返回、用策略表替代长 if/elif 链。")
_rule("long-function", "maintainability", Severity.LOW, "函数过长（>60 行）",
      "按职责拆成多个小函数。")
_rule("too-many-params", "maintainability", Severity.LOW, "参数过多（>5）",
      "聚合为 dataclass/配置对象。")
_rule("deep-nesting", "maintainability", Severity.MEDIUM, "嵌套层级过深（>4）",
      "用卫语句提前返回，或抽取内层逻辑。")
_rule("missing-docstring", "maintainability", Severity.INFO, "公开函数缺少 docstring",
      "补充一行说明用途与参数/返回值。")
_rule("unused-import", "maintainability", Severity.INFO, "导入后未使用",
      "删除未使用的导入，减少启动开销与误读。")

BUILTIN_NAMES = set(dir(builtins))


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _snippet(lines: list[str], lineno: int, span: int = 0) -> str:
    lo = max(1, lineno - span)
    hi = min(len(lines), lineno + span)
    return "\n".join(lines[lo - 1:hi]).strip()


def _is_open_call(node: ast.Call) -> bool:
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "open") or (isinstance(f, ast.Attribute) and f.attr == "open")


class _RuleVisitor(ast.NodeVisitor):
    def __init__(self, sym: Symbol, rel_offset: int, lines: list[str]) -> None:
        self.sym = sym
        self.offset = rel_offset          # 片段内行号 -> 文件绝对行号
        self.lines = lines
        self.parents: dict[int, ast.AST] = {}
        self.out: list[Candidate] = []
        self.tree: ast.AST | None = None

    # ---- 工具 ----
    def abs_line(self, node: ast.AST) -> int:
        return self.offset + getattr(node, "lineno", 1) - 1

    def emit(self, rule_id: str, node: ast.AST, message: str | None = None, span: int = 0) -> None:
        rule = RULES[rule_id]
        rel = getattr(node, "lineno", 1)
        lineno = self.abs_line(node)
        self.out.append(
            Candidate(
                symbol_id=self.sym.id,
                file=self.sym.file,
                lineno=lineno,
                rule_id=rule_id,
                category=rule.category,
                severity=rule.severity,
                title=rule.title,
                message=message or rule.title,
                snippet=_snippet(self.lines, rel, span),   # snippets 用的是符号内相对行号
                suggestion=rule.suggestion,
            )
        )

    def _handler_reraises(self, handler: ast.ExceptHandler) -> bool:
        for sub in ast.walk(handler):
            if isinstance(sub, ast.Raise):
                return True
        return False

    def _in_with(self, node: ast.AST) -> bool:
        cur = self.parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.With, ast.AsyncWith)):
                if any(item.context_expr is node for item in cur.items):
                    return True
            cur = self.parents.get(id(cur))
        return False

    # ---- 访问 ----
    def analyze(self, tree: ast.AST) -> list[Candidate]:
        self.tree = tree
        self.parents = _parents(tree)
        self.visit(tree)
        return self.out

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.emit("bare-except", node)
        else:
            name = ast.unparse(node.type)
            if name in ("Exception", "BaseException") and not self._handler_reraises(node):
                self.emit("broad-except", node)
        body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if all(isinstance(s, ast.Pass) for s in body) or not body:
            self.emit("except-pass", node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # 可变默认参数
        flagged = False
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                flagged = True
            elif isinstance(default, ast.Call) and isinstance(default.func, ast.Name) and default.func.id in {
                "list", "dict", "set",
            }:
                flagged = True
        if flagged:
            self.emit("mutable-default-arg", node)

        # 参数/局部变量遮蔽内置
        arg_names = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
        for name in arg_names & BUILTIN_NAMES:
            self.emit("shadow-builtin", node, message=f"参数 {name} 遮蔽了内置名 {name}")

        # 返回值一致性
        value_returns = [r for r in ast.walk(node) if isinstance(r, ast.Return) and r.value is not None]
        bare_returns = [r for r in ast.walk(node) if isinstance(r, ast.Return) and r.value is None]
        if value_returns and bare_returns:
            self.emit("inconsistent-return", bare_returns[0],
                      message=f"{node.name} 有 {len(value_returns)} 处 return 值、{len(bare_returns)} 处裸 return")

        # 不可达代码
        def walk_body(body: Iterable[ast.stmt]) -> None:
            terminated = False
            for stmt in body:
                if terminated:
                    self.emit("unreachable-code", stmt, span=0)
                    break
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    terminated = True
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, field, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk_body(sub)
                for handler in getattr(stmt, "handlers", []) or []:
                    walk_body(handler.body)

        walk_body(node.body)

        # 复杂度 / 长度 / 参数 / 嵌套
        if self.sym.complexity > 10:
            self.emit("high-complexity", node,
                      message=f"{node.name} 圈复杂度 {self.sym.complexity}（阈值 10）")
        if self.sym.n_lines > 60:
            self.emit("long-function", node, message=f"{node.name} 共 {self.sym.n_lines} 行（阈值 60）")
        if self.sym.n_params > 5:
            self.emit("too-many-params", node, message=f"{node.name} 有 {self.sym.n_params} 个参数（阈值 5）")
        if self.sym.max_nesting > 4:
            self.emit("deep-nesting", node, message=f"{node.name} 最大嵌套 {self.sym.max_nesting} 层（阈值 4）")

        # docstring
        if not node.name.startswith("_") and not ast.get_docstring(node) and len(node.body) > 2:
            self.emit("missing-docstring", node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(comp, ast.Constant) and comp.value is None and isinstance(op, (ast.Eq, ast.NotEq)):
                self.emit("eq-none", node)
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(comp, ast.Constant) and comp.value not in (
                    None, True, False, Ellipsis):
                self.emit("is-literal", node, message=f"用 is 比较字面量 {comp.value!r}")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        seen: set[str] = set()
        for key in node.keys:
            if isinstance(key, ast.Constant):
                k = repr(key.value)
                if k in seen:
                    self.emit("duplicate-dict-key", node, message=f"重复键 {k}")
                    break
                seen.add(k)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        fname = ast.unparse(f)

        # 命令执行 / 动态执行
        if fname in ("os.system", "os.popen"):
            self.emit("os-system", node, message=f"调用 {fname}")
        if isinstance(f, ast.Name) and f.id in ("eval", "exec", "compile"):
            self.emit("eval-exec", node, message=f"调用 {f.id}()")
        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.emit("subprocess-shell-true", node)

        # SQL 拼接
        if isinstance(f, ast.Attribute) and f.attr in ("execute", "executemany", "executescript"):
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.JoinedStr):
                self.emit("sql-fstring", node, message="f-string 直接拼 SQL")
            elif isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Mod, ast.Add)):
                self.emit("sql-fstring", node, message="字符串拼接/格式化构造 SQL")

        # 网络超时
        if fname.startswith(("requests.", "httpx.")) or (
                isinstance(f, ast.Attribute) and f.attr in ("get", "post", "put", "delete", "request")
                and isinstance(f.value, ast.Name) and f.value.id in ("requests", "httpx", "session", "client")):
            if not any(kw.arg == "timeout" for kw in node.keywords):
                self.emit("requests-no-timeout", node, message=f"{fname} 未设置 timeout")

        # yaml / pickle
        if fname in ("yaml.load",):
            if not any(kw.arg == "Loader" for kw in node.keywords):
                self.emit("yaml-unsafe-load", node)
        if fname in ("pickle.load", "pickle.loads"):
            self.emit("pickle-load", node, message=f"调用 {fname}")

        # 未使用 with 的 open
        if _is_open_call(node) and not self._in_with(node):
            self.emit("open-without-with", node)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name_l = target.id.lower()
            if any(k in name_l for k in ("password", "passwd", "secret", "api_key", "apikey", "token")):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) \
                        and len(node.value.value) >= 8:
                    self.emit("hardcoded-secret", node,
                              message=f"变量 {target.id} 直接赋字面量（长度 {len(node.value.value)}）")
            if target.id in BUILTIN_NAMES:
                self.emit("shadow-builtin", node, message=f"变量 {target.id} 遮蔽了内置名")
        self.generic_visit(node)


def analyze_symbol(sym: Symbol) -> list[Candidate]:
    src = sym.source
    if not src.strip():
        return []
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return []
    lines = src.splitlines()
    return _RuleVisitor(sym, sym.lineno, lines).analyze(tree)


def run_rules(index: CodeIndex, symbol_ids: Iterable[str] | None = None) -> list[Candidate]:
    ids = list(symbol_ids) if symbol_ids is not None else list(index.symbols.keys())
    out: list[Candidate] = []
    for sid in ids:
        sym = index.symbols.get(sid)
        if sym is None:
            continue
        # 只分析函数/方法：类的 source 覆盖整个类体，重复扫会把方法问题按"类"再报一遍，
        # 还会让 long-function 用类行数误判。
        if sym.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            continue
        out.extend(analyze_symbol(sym))

    # 文件级：未使用导入
    for file, name, lineno in index.unused_imports:
        if symbol_ids is not None and f"{file}::<module>" not in set(symbol_ids):
            continue
        rule = RULES["unused-import"]
        out.append(
            Candidate(
                symbol_id=f"{file}::<module>",
                file=file,
                lineno=lineno,
                rule_id="unused-import",
                category=rule.category,
                severity=rule.severity,
                title=rule.title,
                message=f"导入 {name} 未使用",
                suggestion=rule.suggestion,
            )
        )
    return out
