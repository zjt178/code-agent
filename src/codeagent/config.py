"""运行配置。所有可调参数集中在此，环境变量优先，其次 .env，最后默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 本机代理常把 127.0.0.1 也劫持走，导致 Ollama 连接被拒；这里强制绕过本地回环。
for _k in ("NO_PROXY", "no_proxy"):
    _hosts = [h for h in os.environ.get(_k, "").split(",") if h]
    for _h in ("127.0.0.1", "localhost", "::1"):
        if _h not in _hosts:
            _hosts.append(_h)
    os.environ[_k] = ",".join(_hosts)

try:  # 可选依赖：没有也能跑
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_backend: str = field(default_factory=lambda: _env("CODEAGENT_LLM", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("CODEAGENT_MODEL", "qwen2.5-coder:7b"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    llm_temperature: float = 0.1
    llm_timeout_s: int = field(default_factory=lambda: _env_int("CODEAGENT_LLM_TIMEOUT", 120))

    # embeddings: auto | ollama | hash
    embed_backend: str = field(default_factory=lambda: _env("CODEAGENT_EMBED", "auto"))
    embed_model: str = field(
        default_factory=lambda: _env("CODEAGENT_EMBED_MODEL", "qllama/bge-small-zh-v1.5:latest")
    )
    hash_dim: int = 512

    # repo guards
    max_files: int = field(default_factory=lambda: _env_int("CODEAGENT_MAX_FILES", 400))
    max_repo_mb: int = field(default_factory=lambda: _env_int("CODEAGENT_MAX_REPO_MB", 60))
    clone_timeout_s: int = field(default_factory=lambda: _env_int("CODEAGENT_CLONE_TIMEOUT", 120))
    max_file_kb: int = 400

    # retrieval
    chunk_top_k: int = 6
    rrf_k: int = 60

    # agent
    max_targets: int = field(default_factory=lambda: _env_int("CODEAGENT_MAX_TARGETS", 8))
    max_tool_steps: int = field(default_factory=lambda: _env_int("CODEAGENT_MAX_TOOL_STEPS", 4))
    max_repair: int = field(default_factory=lambda: _env_int("CODEAGENT_MAX_REPAIR", 2))

    # storage
    storage_dir: Path = field(
        default_factory=lambda: Path(_env("CODEAGENT_STORAGE", str(PROJECT_ROOT / "storage")))
    )


SETTINGS = Settings()
