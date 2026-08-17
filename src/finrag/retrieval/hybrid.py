"""Reciprocal Rank Fusion without access to gold evidence."""

from __future__ import annotations

from finrag.data.schemas import RetrievalHit


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]], top_k: int, rrf_k: int = 60
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievalHit] = {}
    components: dict[str, dict[str, float]] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, 1):
            global_id = hit.chunk.global_id
            scores[global_id] = scores.get(global_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks[global_id] = hit
            component = components.setdefault(global_id, {})
            component.update(hit.component_scores)
            component[f"{hit.method}_rank"] = float(rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [
        RetrievalHit(
            chunk=chunks[key].chunk,
            score=scores[key],
            rank=rank,
            method="hybrid_rrf",
            component_scores={**components[key], "rrf": scores[key]},
        )
        for rank, key in enumerate(ordered, 1)
    ]

