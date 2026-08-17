"""Cross-encoder reranking with an explicit deterministic fallback for CI."""

from __future__ import annotations

import logging
from typing import Any

from finrag.data.schemas import RetrievalHit
from finrag.retrieval.bm25 import tokenize

LOGGER = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", allow_fallback: bool = True) -> None:
        self.model_name = model_name
        self.model: Any | None = None
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name, device=device)
            self.backend = f"cross-encoder:{model_name}"
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Could not load reranker {model_name}") from exc
            LOGGER.warning("Reranker unavailable; using token-overlap fallback: %s", exc)
            self.backend = "fallback:token-overlap"

    def _scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.model is not None:
            logits = self.model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
            scores = []
            for value in logits:
                logit = float(value)
                if logit != logit:
                    raise RuntimeError(
                        "Cross-encoder returned NaN; verify the pinned Transformers version"
                    )
                scores.append(logit)
        else:
            scores = [
                len(set(tokenize(question)) & set(tokenize(content)))
                / max(len(set(tokenize(question))), 1)
                for question, content in pairs
            ]
        return scores

    @staticmethod
    def _rank(hits: list[RetrievalHit], scores: list[float], top_k: int) -> list[RetrievalHit]:
        pairs = sorted(zip(hits, scores, strict=True), key=lambda pair: (-pair[1], pair[0].rank))[
            :top_k
        ]
        return [
            RetrievalHit(
                chunk=hit.chunk,
                score=float(score),
                rank=rank,
                method="hybrid_rerank",
                component_scores={**hit.component_scores, "reranker": float(score)},
            )
            for rank, (hit, score) in enumerate(pairs, 1)
        ]

    def rerank(self, question: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        scores = self._scores([(question, hit.chunk.content) for hit in hits])
        return self._rank(hits, scores, top_k)

    def rerank_many(
        self,
        questions: list[str],
        hit_lists: list[list[RetrievalHit]],
        top_k: int,
    ) -> list[list[RetrievalHit]]:
        """Batch independent query-passage pairs for efficient offline evaluation."""
        if len(questions) != len(hit_lists):
            raise ValueError("questions and hit_lists must have equal length")
        flat_pairs = [
            (question, hit.chunk.content)
            for question, hits in zip(questions, hit_lists, strict=True)
            for hit in hits
        ]
        flat_scores = self._scores(flat_pairs) if flat_pairs else []
        results: list[list[RetrievalHit]] = []
        offset = 0
        for hits in hit_lists:
            count = len(hits)
            results.append(self._rank(hits, flat_scores[offset : offset + count], top_k))
            offset += count
        return results
