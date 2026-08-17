"""Unified retrieval index used by the CLI, graph, evaluation, and UI."""

from __future__ import annotations

import time
from collections.abc import Collection

from finrag.config import RetrievalConfig
from finrag.data.schemas import DocumentChunk, RetrievalHit
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.hybrid import reciprocal_rank_fusion
from finrag.retrieval.reranker import CrossEncoderReranker


class RetrievalIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        config: RetrievalConfig,
        allow_model_fallback: bool = True,
    ) -> None:
        self.chunks = chunks
        self.config = config
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(
            chunks,
            model_name=config.embedding_model,
            device=config.device,
            allow_fallback=allow_model_fallback,
        )
        self.reranker = CrossEncoderReranker(
            config.reranker_model,
            device=config.device,
            allow_fallback=allow_model_fallback,
        )
        self.last_timing_ms: dict[str, float] = {}

    @property
    def backends(self) -> dict[str, str]:
        return {"dense": self.dense.backend, "reranker": self.reranker.backend}

    def search(
        self,
        question: str,
        method: str | None = None,
        top_k: int | None = None,
        allowed_report_ids: Collection[str] | None = None,
    ) -> list[RetrievalHit]:
        selected_method = method or self.config.method
        limit = top_k or self.config.top_k
        self.last_timing_ms = {}
        started = time.perf_counter()
        if selected_method == "bm25":
            hits = self.bm25.search(question, limit, allowed_report_ids)
            self.last_timing_ms["retrieval"] = (time.perf_counter() - started) * 1000
            return hits
        if selected_method == "dense":
            hits = self.dense.search(question, limit, allowed_report_ids)
            self.last_timing_ms["retrieval"] = (time.perf_counter() - started) * 1000
            return hits
        lexical = self.bm25.search(question, self.config.candidate_k, allowed_report_ids)
        dense = self.dense.search(question, self.config.candidate_k, allowed_report_ids)
        fused = reciprocal_rank_fusion(
            [lexical, dense], top_k=self.config.candidate_k, rrf_k=self.config.rrf_k
        )
        self.last_timing_ms["retrieval"] = (time.perf_counter() - started) * 1000
        if selected_method == "hybrid":
            return fused[:limit]
        if selected_method != "hybrid_rerank":
            raise ValueError(f"Unknown retrieval method: {selected_method}")
        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(question, fused, top_k=limit)
        self.last_timing_ms["reranking"] = (time.perf_counter() - rerank_started) * 1000
        return reranked
