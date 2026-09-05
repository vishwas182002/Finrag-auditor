"""Paired comparison of retrieval runs evaluated on identical question IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finrag.evaluation.bootstrap import bootstrap_mean_ci
from finrag.evaluation.retrieval_metrics import METRIC_VERSION

METRICS = ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5")


def _method_rows(path: Path, method: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["method"] == method:
            if row.get("metric_version") != METRIC_VERSION:
                raise ValueError(
                    "Retrieval comparison requires report-scoped-v2 rows; regrade or rerun legacy artifacts"
                )
            if str(row["question_id"]) in rows:
                raise ValueError("Duplicate question ID in retrieval rows")
            rows[str(row["question_id"])] = row
    if not rows:
        raise ValueError(f"No {method} rows found in {path}")
    return rows


def compare_retrieval_runs(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    method: str = "hybrid_rerank",
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    baseline = _method_rows(baseline_path, method)
    candidate = _method_rows(candidate_path, method)
    if set(baseline) != set(candidate):
        raise ValueError("Retrieval runs do not contain identical question IDs")
    question_ids = sorted(baseline)
    report: dict[str, Any] = {
        "method": method,
        "metric_version": METRIC_VERSION,
        "question_count": len(question_ids),
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "paired_deltas_candidate_minus_baseline": {},
    }
    for metric in METRICS:
        differences = [
            float(candidate[question_id][metric]) - float(baseline[question_id][metric])
            for question_id in question_ids
        ]
        report["paired_deltas_candidate_minus_baseline"][metric] = {
            "mean": sum(differences) / len(differences),
            "bootstrap_95_ci": bootstrap_mean_ci(differences, bootstrap_samples, seed),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
