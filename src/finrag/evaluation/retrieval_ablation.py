"""Retrieval-only ablation for development-set model selection."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from finrag.config import AppConfig
from finrag.data.finqa import load_finqa, select_configured, selected_corpus_examples
from finrag.data.schemas import RetrievalHit
from finrag.evaluation.bootstrap import bootstrap_mean_ci
from finrag.evaluation.retrieval_metrics import mean_metrics, retrieval_metrics
from finrag.indexing.chunking import build_chunks
from finrag.indexing.index import RetrievalIndex

LOGGER = logging.getLogger(__name__)


def run_retrieval_ablation(config: AppConfig, project_root: Path) -> dict[str, Any]:
    examples = load_finqa(config.data.path)
    selected = select_configured(
        examples,
        config.data.sample_size,
        config.seed,
        config.data.question_manifest,
    )
    corpus = selected_corpus_examples(examples, selected, config.data.corpus_scope)
    chunks = build_chunks(corpus, config.chunking)
    index = RetrievalIndex(chunks, config.retrieval)
    methods = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    report: dict[str, Any] = {
        "metadata": {
            "split": config.data.split,
            "sample_size": len(selected),
            "corpus_scope": config.data.corpus_scope,
            "corpus_chunks": len(chunks),
            "backends": index.backends,
            "gold_evidence_used_only_for_scoring": True,
            "generation_model_called": False,
        },
        "methods": {},
    }
    detail_rows: list[dict[str, Any]] = []
    for method in methods:
        rows: list[dict[str, float]] = []
        latencies: list[float] = []
        if method == "hybrid_rerank":
            candidate_lists: list[list[RetrievalHit]] = []
            retrieval_latencies: list[float] = []
            for position, example in enumerate(selected, 1):
                started = time.perf_counter()
                candidate_lists.append(
                    index.search(
                        example.question,
                        method="hybrid",
                        top_k=config.retrieval.candidate_k,
                    )
                )
                retrieval_latencies.append((time.perf_counter() - started) * 1000)
                if position % 100 == 0 or position == len(selected):
                    LOGGER.info(
                        "Retrieval ablation %s candidates: %d/%d questions",
                        method,
                        position,
                        len(selected),
                    )
            rerank_started = time.perf_counter()
            hit_lists = index.reranker.rerank_many(
                [example.question for example in selected],
                candidate_lists,
                top_k=5,
            )
            rerank_total_ms = (time.perf_counter() - rerank_started) * 1000
            amortized_rerank_ms = rerank_total_ms / max(len(selected), 1)
            latencies = [value + amortized_rerank_ms for value in retrieval_latencies]
            for example, hits in zip(selected, hit_lists, strict=True):
                metrics = retrieval_metrics(hits, example.gold_source_ids)
                rows.append(metrics)
                detail_rows.append(
                    {
                        "question_id": example.question_id,
                        "method": method,
                        **metrics,
                    }
                )
            LOGGER.info(
                "Retrieval ablation %s batched reranking complete: %.1f ms total",
                method,
                rerank_total_ms,
            )
        else:
            for position, example in enumerate(selected, 1):
                started = time.perf_counter()
                hits = index.search(example.question, method=method, top_k=5)
                latencies.append((time.perf_counter() - started) * 1000)
                metrics = retrieval_metrics(hits, example.gold_source_ids)
                rows.append(metrics)
                detail_rows.append(
                    {
                        "question_id": example.question_id,
                        "method": method,
                        **metrics,
                    }
                )
                if position % 100 == 0 or position == len(selected):
                    LOGGER.info(
                        "Retrieval ablation %s: %d/%d questions",
                        method,
                        position,
                        len(selected),
                    )
        aggregate = mean_metrics(rows)
        report["methods"][method] = {
            "metrics": aggregate,
            "bootstrap_95_ci": {
                metric: bootstrap_mean_ci(
                    [row[metric] for row in rows],
                    config.evaluation.bootstrap_samples,
                    config.seed,
                )
                for metric in aggregate
            },
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }
        if method == "hybrid_rerank":
            report["methods"][method]["latency_note"] = (
                "Offline batched reranker time amortized across questions; not online latency."
            )
    output_dir = project_root / "artifacts" / "retrieval_ablation" / config.evaluation.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "retrieval_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return report
