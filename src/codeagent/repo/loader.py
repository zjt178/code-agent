"""仓库装载：Git URL 浅克隆 / 本地目录，带体积与文件数护栏。

安全边界：只读取文件，不执行仓库内任何代码（不 import、不安装依赖、不跑测试）。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import SETTINGS

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "env", "node_modules", "site-packages", "build", "dist",
    ".eggs", "migrations", ".idea", ".vscode",
}


@dataclass
class LoadedRepo:
    root: Path
    display: str
    commit: str | None
    py_files: list[Path]
    rel_files: list[str]
    skipped: list[str]
    truncated: bool

    @property
    def repo_id(self) -> str:
        h = hashlib.sha1(f"{self.display}:{self.commit}".encode()).hexdigest()[:10]
        return h


class RepoLoadError(RuntimeError):
    pass


def _is_local(target: str) -> bool:
    p = Path(target).expanduser()
    return p.exists()


def _looks_like_url(target: str) -> bool:
    return target.startswith(("http://", "https://", "git@", "ssh://"))


def load_repo(
    target: str,
    ref: str | None = None,
    subdir: str | None = None,
    workdir: Path | None = None,
) -> LoadedRepo:
    """把目标仓库准备成磁盘目录，并列出待分析的 Python 文件。"""
    if _is_local(target):
        root = Path(target).expanduser().resolve()
        commit = _local_commit(root)
        display = str(root)
        if subdir:
            root = root / subdir
        if not root.exists():
            raise RepoLoadError(f"子目录不存在：{subdir}")
        return _scan(root, display, commit)

    if not _looks_like_url(target):
        raise RepoLoadError(f"既不是本地路径也不是 Git URL：{target}")

    dest_parent = workdir or Path(tempfile.mkdtemp(prefix="codeagent-"))
    dest = Path(dest_parent) / "repo"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    try:
        from git import Repo  # GitPython
    except Exception as exc:  # pragma: no cover
        raise RepoLoadError("缺少 GitPython，请先 pip install -r requirements.txt") from exc

    t0 = time.time()
    errors: list[str] = []
    for attempt, env in enumerate(_git_envs()):
        try:
            repo = Repo.clone_from(
                target,
                dest,
                depth=1,
                single_branch=True,
                no_tags=True,
                multi_options=["--filter=blob:none"] if ref is None else None,
                env=env,
            )
            if ref:
                repo.git.fetch("--depth", "1", "origin", ref)
                repo.git.checkout("FETCH_HEAD")
            break
        except Exception as exc:
            errors.append(f"第 {attempt + 1} 次：{str(exc)[:200]}")
            shutil.rmtree(dest, ignore_errors=True)
    else:
        raise RepoLoadError(
            f"克隆失败（{time.time() - t0:.1f}s）：" + "；".join(errors[-2:])
        )

    root = dest / subdir if subdir else dest
    if not root.exists():
        raise RepoLoadError(f"子目录不存在：{subdir}")
    return _scan(root, target, _local_commit(dest))


def _git_envs() -> list[dict[str, str]]:
    """三套 git 环境，依次尝试：继承环境 → 去掉代理环境变量 → 连 git 全局 http.proxy 也清掉。

    实测：本机 git 全局配置了 http.proxy=127.0.0.1:7897，代理一停，clone 就报
    "Failed to connect ... over proxy"，而 GitHub 直连是通的。只清环境变量不够，
    必须用 GIT_CONFIG_* 覆盖配置项。
    """
    base = dict(os.environ)
    base["GIT_TERMINAL_PROMPT"] = "0"

    no_proxy_env = {k: v for k, v in base.items()
                    if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}

    clear_proxy_cfg = dict(no_proxy_env)
    clear_proxy_cfg.update({
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.proxy", "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": "https.proxy", "GIT_CONFIG_VALUE_1": "",
    })
    return [base, no_proxy_env, clear_proxy_cfg]


def _local_commit(root: Path) -> str | None:
    try:
        from git import Repo

        return Repo(root, search_parent_directories=True).head.commit.hexsha[:10]
    except Exception:
        return None


def _scan(root: Path, display: str, commit: str | None) -> LoadedRepo:
    py_files: list[Path] = []
    skipped: list[str] = []
    total_bytes = 0
    truncated = False
    limit_bytes = SETTINGS.max_repo_mb * 1024 * 1024

    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > SETTINGS.max_file_kb * 1024:
            skipped.append(f"{rel.as_posix()}（{size // 1024}KB 超单文件上限）")
            continue
        total_bytes += size
        if total_bytes > limit_bytes:
            truncated = True
            break
        py_files.append(path)
        if len(py_files) >= SETTINGS.max_files:
            truncated = True
            break

    if not py_files:
        raise RepoLoadError("未找到可分析的 Python 文件")

    rels = [p.relative_to(root).as_posix() for p in py_files]
    return LoadedRepo(
        root=root,
        display=display,
        commit=commit,
        py_files=py_files,
        rel_files=rels,
        skipped=skipped,
        truncated=truncated,
    )
