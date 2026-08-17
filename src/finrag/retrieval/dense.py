"""Sentence-transformer dense retrieval with an explicit CI fallback."""

from __future__ import annotations

import logging
import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from finrag.data.schemas import DocumentChunk, RetrievalHit

LOGGER = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        model_name: str,
        device: str = "cpu",
        allow_fallback: bool = True,
    ) -> None:
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk")
        self.chunks = chunks
        self.model_name = model_name
        self.device = device
        self.model: Any | None = None
        self.vectorizer: HashingVectorizer | None = None
        texts = [chunk.content for chunk in chunks]
        try:
            os.environ.setdefault(
                "HF_HOME", str(Path(__file__).resolve().parents[3] / ".hf_cache")
            )
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)
            self.embeddings = np.asarray(
                self.model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            self.backend = f"sentence-transformers:{model_name}"
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Could not load dense model {model_name}") from exc
            LOGGER.warning("Dense model unavailable; using named hashing fallback: %s", exc)
            self.vectorizer = HashingVectorizer(
                n_features=2048, alternate_sign=False, norm=None, ngram_range=(1, 2)
            )
            self.embeddings = normalize(self.vectorizer.transform(texts)).toarray().astype(np.float32)
            self.backend = "fallback:sklearn-hashing-2048"

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            return np.asarray(
                self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
                dtype=np.float32,
            )
        assert self.vectorizer is not None
        return normalize(self.vectorizer.transform([query])).toarray()[0].astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int,
        allowed_report_ids: Collection[str] | None = None,
    ) -> list[RetrievalHit]:
        query_vector = self._encode_query(query)
        scores = self.embeddings @ query_vector
        allowed = set(allowed_report_ids) if allowed_report_ids is not None else None
        candidates = [
            i for i, chunk in enumerate(self.chunks) if allowed is None or chunk.report_id in allowed
        ]
        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.chunks[i].global_id))[
            :top_k
        ]
        return [
            RetrievalHit(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank,
                method="dense",
                component_scores={"dense": float(scores[i])},
            )
            for rank, i in enumerate(ordered, 1)
        ]
