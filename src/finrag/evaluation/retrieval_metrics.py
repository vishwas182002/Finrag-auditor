"""Retrieval metrics computed against FinQA supporting-evidence annotations."""

from __future__ import annotations

import math
from collections.abc import Iterable

from finrag.data.schemas import RetrievalHit


def relevance_vector(hits: list[RetrievalHit], gold_source_ids: set[str]) -> list[int]:
    return [int(bool(set(hit.chunk.source_ids) & gold_source_ids)) for hit in hits]


def retrieval_metrics(hits: list[RetrievalHit], gold_source_ids: Iterable[str]) -> dict[str, float]:
    gold = set(gold_source_ids)
    relevance = relevance_vector(hits, gold)
    metrics: dict[str, float] = {}
    for k in (1, 3, 5):
        retrieved_sources = {
            source_id for hit in hits[:k] for source_id in hit.chunk.source_ids
        }
        metrics[f"recall@{k}"] = len(retrieved_sources & gold) / max(len(gold), 1)
    relevant_ranks = [rank for rank, value in enumerate(relevance, 1) if value]
    metrics["mrr"] = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance[:5], 1))
    ideal_count = min(len(gold), 5)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    metrics["ndcg@5"] = dcg / idcg if idcg else 0.0
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }

