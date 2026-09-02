"""BM25 lexical retrieval with metadata-preserving results."""

from __future__ import annotations

import re
from collections.abc import Collection

import numpy as np
from rank_bm25 import BM25Okapi

from finrag.data.schemas import DocumentChunk, RetrievalHit

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self.chunks = chunks
        self._index = BM25Okapi([tokenize(chunk.content) for chunk in chunks])

    def search(
        self,
        query: str,
        top_k: int,
        allowed_report_ids: Collection[str] | None = None,
    ) -> list[RetrievalHit]:
        scores = np.asarray(self._index.get_scores(tokenize(query)), dtype=float)
        allowed = set(allowed_report_ids) if allowed_report_ids is not None else None
        candidates = [
            i for i, chunk in enumerate(self.chunks) if allowed is None or chunk.report_id in allowed
        ]
        ordered = sorted(candidates, key=lambda i: (-scores[i], self.chunks[i].global_id))[:top_k]
        return [
            RetrievalHit(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank,
                method="bm25",
                component_scores={"bm25": float(scores[i])},
            )
            for rank, i in enumerate(ordered, 1)
        ]

