"""混合检索单测：BM25 命中标识符；RRF 融合两路结果；排除自身。"""
from __future__ import annotations

from pathlib import Path

from codeagent.repo.ast_index import CodeIndex
from codeagent.retrieval.embed import HashEmbedder
from codeagent.retrieval.hybrid import HybridRetriever

SRC = '''
def parse_yaml(text):
    """把 yaml 文本解析为字典。"""
    return {"raw": text}


def send_email(to, subject):
    """发送邮件。"""
    return f"{to}:{subject}"


def compute_tax(amount):
    """计算税费。"""
    return amount * 0.13
'''


def _retriever(tmp_path: Path) -> HybridRetriever:
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    idx = CodeIndex().build(tmp_path, ["m.py"])
    return HybridRetriever(idx.to_chunks(), HashEmbedder(), "test-hash")


def test_bm25_finds_identifier(tmp_path: Path) -> None:
    r = _retriever(tmp_path)
    hits = r.search("parse_yaml", k=2, mode="bm25")
    assert hits and hits[0].chunk.symbol_id.endswith("parse_yaml")


def test_hybrid_returns_sources(tmp_path: Path) -> None:
    r = _retriever(tmp_path)
    hits = r.search("税费 计算 amount", k=3)
    assert hits
    assert all(h.sources for h in hits)


def test_exclude_filter(tmp_path: Path) -> None:
    r = _retriever(tmp_path)
    hits = r.search("parse_yaml", k=3, exclude={"m.py::parse_yaml"})
    assert all(h.chunk.symbol_id != "m.py::parse_yaml" for h in hits)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    r = _retriever(tmp_path)
    out = tmp_path / "idx"
    r.save(out)
    loaded = HybridRetriever.load(out, HashEmbedder())
    hits = loaded.search("发送邮件", k=2)
    assert hits and hits[0].chunk.symbol_id.endswith("send_email")
