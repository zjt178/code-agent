"""Pydantic 数据契约：全项目共享的输入/输出结构。

设计原则：
- 所有中间产物都是可序列化对象，便于落盘、API 返回、MCP 复用；
- 缺陷（Finding）与候选（Candidate）分离：候选是规则命中，Finding 是经 LLM 复核后的结论，
  这样"规则召回率"与"复核准确率"可以分别度量。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Symbol(BaseModel):
    """一个可被检索/分析的代码符号（模块/类/函数/方法）。"""

    id: str                      # "src/foo.py::Bar.baz"
    file: str                    # 相对仓库根的 posix 路径
    qualname: str                # "Bar.baz"
    name: str
    kind: SymbolKind
    lineno: int
    end_lineno: int
    n_params: int = 0
    required_params: int = 0
    complexity: int = 1
    max_nesting: int = 0
    is_async: bool = False
    decorators: list[str] = Field(default_factory=list)
    bases: list[str] = Field(default_factory=list)
    docstring: str | None = None
    parent: str | None = None
    source: str = ""
    callees: list[str] = Field(default_factory=list)

    @property
    def n_lines(self) -> int:
        return max(0, self.end_lineno - self.lineno + 1)


class Chunk(BaseModel):
    """检索单元：函数级切片，带路径/类名等结构化前缀。"""

    id: str
    symbol_id: str
    file: str
    lineno: int
    end_lineno: int
    kind: SymbolKind
    text: str
    tokens: int = 0


class ImportRef(BaseModel):
    module: str
    name: str | None = None
    alias: str | None = None
    lineno: int = 1


class ParseError(BaseModel):
    file: str
    message: str
    lineno: int | None = None


class RepoStats(BaseModel):
    repo: str
    commit: str | None = None
    files: int = 0
    symbols: int = 0
    functions: int = 0
    classes: int = 0
    loc: int = 0
    chunks: int = 0
    parse_errors: list[ParseError] = Field(default_factory=list)


class Candidate(BaseModel):
    """规则引擎命中的候选缺陷（未复核）。"""

    symbol_id: str
    file: str
    lineno: int
    rule_id: str
    category: str
    severity: Severity
    title: str
    message: str
    snippet: str = ""
    suggestion: str = ""


class Finding(BaseModel):
    """经复核后的缺陷结论。"""

    id: str
    symbol_id: str
    file: str
    lineno: int
    rule_id: str
    category: str
    severity: Severity
    title: str
    detail: str
    suggestion: str = ""
    confidence: float = 0.5
    confirmed_by: Literal["rule", "llm"] = "rule"
    evidence: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    """生成的测试用例（静态产物，本项目不执行）。"""

    __test__ = False  # 防止 pytest 误当成测试类收集

    id: str
    target_symbol_id: str
    target: str                 # 测试函数名
    file: str                   # 被测源文件
    code: str                   # 完整测试函数源码
    focus: Literal["happy", "edge", "error"] = "happy"
    status: Literal["valid", "warn", "rejected"] = "valid"
    issues: list[str] = Field(default_factory=list)
    generator: Literal["llm", "template"] = "template"


class TestModule(BaseModel):
    __test__ = False

    path: str
    code: str
    cases: list[TestCase] = Field(default_factory=list)


class CheckIssue(BaseModel):
    level: Literal["error", "warn"]
    code: str                      # SYNTAX / UNKNOWN_SYMBOL / SIDE_EFFECT / NO_ASSERT ...
    message: str
    test: str | None = None


class CheckReport(BaseModel):
    ok: bool = True
    issues: list[CheckIssue] = Field(default_factory=list)
    checked_tests: int = 0
    valid_tests: int = 0
    coverage_ratio: float = 0.0     # 有测试的目标函数 / 应有测试的目标函数
    untested_targets: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """Agent 在 investigate 阶段收集的证据片段。"""

    symbol_id: str
    kind: Literal["self", "callee", "caller", "similar", "rule"] = "similar"
    reason: str = ""
    snippet: str = ""
    score: float = 0.0


class TargetPlan(BaseModel):
    symbol_id: str
    file: str
    qualname: str
    risk_score: float
    reasons: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    node: str
    action: str
    detail: str = ""
    ms: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    repo: str = Field(description="Git URL 或本地路径")
    ref: str | None = Field(default=None, description="分支/commit，默认默认分支")
    subdir: str | None = Field(default=None, description="只分析仓库子目录")
    max_targets: int = 8
    use_llm: bool = True
    generate_tests: bool = True
    max_repair: int = 2


class Report(BaseModel):
    job_id: str
    repo: str
    stats: RepoStats
    targets: list[TargetPlan] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    tests: list[TestModule] = Field(default_factory=list)
    check: CheckReport = Field(default_factory=CheckReport)
    candidates_total: int = 0
    llm_calls: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: list[TraceEvent] = Field(default_factory=list)
    llm_enabled: bool = False
    notes: list[str] = Field(default_factory=list)
