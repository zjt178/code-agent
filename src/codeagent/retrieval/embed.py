"""向量化后端。

两种实现：
- OllamaEmbedder：调用本机模型（默认 bge-small-zh），需要 ollama serve；
- HashEmbedder：字符/词哈希 + 次线性 TF + L2 归一化，纯离线确定性，保证零依赖可跑。

接口相同，上层索引不感知差异；切换用环境变量 CODEAGENT_EMBED=ollama|hash|auto。
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence

import numpy as np

from ..config import SETTINGS
from ..repo.ast_index import tokenize_code


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashEmbedder:
    """确定性哈希向量：同样的输入永远同样的输出，便于离线复现实验结果。"""

    def __init__(self, dim: int = SETTINGS.hash_dim) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[int, int] = {}
        for tok in tokenize_code(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            counts[idx] = counts.get(idx, 0) + 1
        for idx, c in counts.items():
            vec[idx] = 1.0 + np.log(c)
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)


class OllamaEmbedder:
    def __init__(self, model: str = SETTINGS.embed_model, base_url: str = SETTINGS.ollama_base_url) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = f"ollama:{model}"
        # 显式禁用代理：本机代理会把 127.0.0.1 也劫持走，导致 404/连接被拒
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.dim = self._probe_dim()

    def _post(self, payload: dict, timeout: int = 30) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _probe_dim(self) -> int:
        out = self._post({"model": self.model, "input": ["def f(x): return x"]}, timeout=30)
        emb = out.get("embeddings") or []
        if not emb:
            raise RuntimeError("ollama 未返回 embedding")
        return len(emb[0])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._post({"model": self.model, "input": list(texts)}, timeout=180)
        arr = np.array(out["embeddings"], dtype=np.float32)
        return _l2(arr)


def ollama_available(base_url: str = SETTINGS.ollama_base_url, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def build_embedder(backend: str | None = None) -> tuple[Embedder, str]:
    """返回 (embedder, note)。note 说明降级原因，写进报告便于复现。"""
    backend = (backend or SETTINGS.embed_backend).lower()
    if backend in ("auto", "ollama") and ollama_available():
        try:
            return OllamaEmbedder(), f"embedding=ollama:{SETTINGS.embed_model}"
        except Exception as exc:
            if backend == "ollama":
                raise
            return HashEmbedder(), f"embedding 降级为 hash（ollama 调用失败：{exc}）"
    if backend == "ollama":
        raise RuntimeError("CODEAGENT_EMBED=ollama 但本机 ollama 未启动")
    return HashEmbedder(), "embedding=hash（未检测到 ollama，语义检索退化为词法哈希）"
