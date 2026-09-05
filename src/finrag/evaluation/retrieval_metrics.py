"""Retrieval metrics computed against FinQA supporting-evidence annotations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from finrag.data.schemas import FinQAExample, RetrievalHit

METRIC_VERSION = "report-scoped-v2"


def relevance_vector(
    hits: list[RetrievalHit], gold_source_ids: set[str], gold_report_id: str
) -> list[int]:
    return [
        int(
            hit.chunk.report_id == gold_report_id
            and bool(set(hit.chunk.source_ids) & gold_source_ids)
        )
        for hit in hits
    ]


def retrieval_metrics(
    hits: list[RetrievalHit], gold_source_ids: Iterable[str], gold_report_id: str
) -> dict[str, float]:
    gold = set(gold_source_ids)
    relevance = relevance_vector(hits, gold, gold_report_id)
    metrics: dict[str, float] = {}
    for k in (1, 3, 5):
        retrieved_sources = {
            source_id
            for hit in hits[:k]
            if hit.chunk.report_id == gold_report_id
            for source_id in hit.chunk.source_ids
        }
        metrics[f"recall@{k}"] = len(retrieved_sources & gold) / max(len(gold), 1)
    relevant_ranks = [rank for rank, value in enumerate(relevance, 1) if value]
    metrics["mrr"] = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    # Binary novel-evidence gain: overlapping chunks cannot repeatedly earn credit
    # for one gold source, or make nDCG exceed 1.
    seen: set[str] = set()
    dcg = 0.0
    for rank, hit in enumerate(hits[:5], 1):
        sources = (
            set(hit.chunk.source_ids) & gold if hit.chunk.report_id == gold_report_id else set()
        )
        if sources - seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.update(sources)
    ideal_count = min(len(gold), 5)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    metrics["ndcg@5"] = dcg / idcg if idcg else 0.0
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def retrieval_record(
    example: FinQAExample, hits: list[RetrievalHit], method: str, metrics: dict[str, float]
) -> dict[str, Any]:
    """Persist enough evidence to independently regrade every retrieval method."""
    return {
        "question_id": example.question_id,
        "report_id": example.report_id,
        "gold_source_ids": list(example.gold_source_ids),
        "method": method,
        "metric_version": METRIC_VERSION,
        "retrieved": [
            {
                "report_id": hit.chunk.report_id,
                "citation_id": hit.chunk.citation_id,
                "source_ids": list(hit.chunk.source_ids),
                "rank": hit.rank,
            }
            for hit in hits
        ],
        **metrics,
    }
