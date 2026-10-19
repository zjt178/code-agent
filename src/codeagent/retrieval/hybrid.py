"""混合检索：BM25（词法）+ FAISS（语义）+ RRF 融合。

RRF 只用排名不用分数，因此两路分数量纲不同也不会互相压制——这一点在项目一里已经验证过，
这里直接复用同一套融合口径，保证两个项目的评估结论可比。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from rank_bm25 import BM25Plus

from ..config import SETTINGS
from ..repo.ast_index import tokenize_code
from ..schemas import Chunk
from .embed import Embedder, build_embedder

Mode = Literal["hybrid", "vector", "bm25"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: tuple[str, ...] = ()

    @property
    def symbol_id(self) -> str:
        return self.chunk.symbol_id

    def as_dict(self) -> dict:
        return {
            "symbol_id": self.symbol_id,
            "file": self.chunk.file,
            "lineno": self.chunk.lineno,
            "score": round(self.score, 5),
            "sources": list(self.sources),
            "preview": self.chunk.text[:280],
        }


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, note: str = "") -> None:
        self.chunks = chunks
        self.embedder = embedder or build_embedder()[0]
        self.note = note
        self._build()

    def _build(self) -> None:
        self.tokens = [tokenize_code(c.text) for c in self.chunks]
        self.bm25 = BM25Plus(self.tokens) if self.tokens else None
        vecs = self.embedder.embed([c.text for c in self.chunks]) if self.chunks else np.zeros((0, 1), np.float32)
        self.vectors = vecs
        if len(vecs):
            self.faiss = faiss.IndexFlatIP(vecs.shape[1])
            self.faiss.add(vecs.astype(np.float32))
        else:
            self.faiss = None

    # ---------- 单路检索 ----------
    def _bm25_hits(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize_code(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 1e-9]

    def _vector_hits(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.faiss is None:
            return []
        q = self.embedder.embed([query])
        sims, idx = self.faiss.search(q.astype(np.float32), min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0]

    # ---------- 融合检索 ----------
    def search(self, query: str, k: int = SETTINGS.chunk_top_k, mode: Mode = "hybrid",
               exclude: set[str] | None = None) -> list[Hit]:
        exclude = exclude or set()
        pool = max(k * 4, 12)
        if mode == "bm25":
            ranked = [("bm25", self._bm25_hits(query, pool))]
        elif mode == "vector":
            ranked = [("vector", self._vector_hits(query, pool))]
        else:
            ranked = [("bm25", self._bm25_hits(query, pool)), ("vector", self._vector_hits(query, pool))]

        fused: dict[int, float] = {}
        srcs: dict[int, set[str]] = {}
        for name, hits in ranked:
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (SETTINGS.rrf_k + rank + 1)
                srcs.setdefault(idx, set()).add(name)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        out: list[Hit] = []
        for idx, score in ordered:
            chunk = self.chunks[idx]
            if chunk.symbol_id in exclude:
                continue
            out.append(Hit(chunk=chunk, score=score, sources=tuple(sorted(srcs[idx]))))
            if len(out) >= k:
                break
        return out

    # ---------- 落盘 ----------
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        np.save(out_dir / "vectors.npy", self.vectors)
        (out_dir / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "dim": int(self.vectors.shape[1]) if len(self.vectors) else 0,
                        "chunks": len(self.chunks), "note": self.note}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, in_dir: Path, embedder: Embedder | None = None) -> "HybridRetriever":
        chunks = [Chunk(**d) for d in json.loads((in_dir / "chunks.json").read_text(encoding="utf-8"))]
        emb = embedder or build_embedder()[0]
        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.embedder = emb
        obj.note = f"loaded:{in_dir.name}"
        obj.tokens = [tokenize_code(c.text) for c in chunks]
        obj.bm25 = BM25Plus(obj.tokens) if obj.tokens else None
        vectors = np.load(in_dir / "vectors.npy")
        if vectors.shape[0] != len(chunks) or (len(vectors) and vectors.shape[1] != emb.dim):
            obj.vectors = emb.embed([c.text for c in chunks])
        else:
            obj.vectors = vectors
        if len(obj.vectors):
            obj.faiss = faiss.IndexFlatIP(obj.vectors.shape[1])
            obj.faiss.add(obj.vectors.astype(np.float32))
        else:
            obj.faiss = None
        return obj
