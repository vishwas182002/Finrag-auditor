from __future__ import annotations

import sentence_transformers

from finrag.data.schemas import DocumentChunk, RetrievalHit
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.hybrid import reciprocal_rank_fusion
from finrag.retrieval.reranker import CrossEncoderReranker


def test_bm25_retrieval(chunks: list[DocumentChunk]) -> None:
    hits = BM25Retriever(chunks).search("2023 revenue", top_k=2)
    assert hits[0].chunk.chunk_id == "table_1"
    assert hits[0].method == "bm25"


def test_dense_retrieval_interface_uses_explicit_fallback(monkeypatch, chunks) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", unavailable)
    retriever = DenseRetriever(chunks, "not-downloaded", allow_fallback=True)
    hits = retriever.search("revenue", top_k=2)
    assert len(hits) == 2
    assert retriever.backend == "fallback:sklearn-hashing-2048"
    assert all(hit.method == "dense" for hit in hits)


def test_reciprocal_rank_fusion(chunks: list[DocumentChunk]) -> None:
    lexical = [RetrievalHit(chunk=chunks[0], score=4, rank=1, method="bm25")]
    dense = [
        RetrievalHit(chunk=chunks[1], score=0.8, rank=1, method="dense"),
        RetrievalHit(chunk=chunks[0], score=0.7, rank=2, method="dense"),
    ]
    fused = reciprocal_rank_fusion([lexical, dense], top_k=2, rrf_k=60)
    assert fused[0].chunk.global_id == chunks[0].global_id
    assert fused[0].component_scores["bm25_rank"] == 1.0
    assert fused[0].component_scores["dense_rank"] == 2.0


def test_reranker_interface_uses_explicit_fallback(monkeypatch, hits) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", unavailable)
    reranker = CrossEncoderReranker("not-downloaded", allow_fallback=True)
    ranked = reranker.rerank("change in revenue", hits, top_k=1)
    assert ranked[0].method == "hybrid_rerank"
    assert "reranker" in ranked[0].component_scores
    batched = reranker.rerank_many(
        ["change in revenue", "revenue difference"], [hits, hits], top_k=1
    )
    assert [group[0].chunk.global_id for group in batched] == [
        hits[0].chunk.global_id,
        hits[0].chunk.global_id,
    ]
