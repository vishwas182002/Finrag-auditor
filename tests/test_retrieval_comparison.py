from __future__ import annotations

import json

from finrag.evaluation.retrieval_comparison import compare_retrieval_runs


def test_paired_retrieval_comparison(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "comparison.json"
    baseline_rows = [
        {
            "question_id": question_id,
            "method": "hybrid_rerank",
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
        }
        for question_id in ("q1", "q2")
    ]
    candidate_rows = [{**row, "mrr": 1.0} for row in baseline_rows]
    baseline.write_text("".join(json.dumps(row) + "\n" for row in baseline_rows))
    candidate.write_text("".join(json.dumps(row) + "\n" for row in candidate_rows))
    report = compare_retrieval_runs(baseline, candidate, output, bootstrap_samples=10)
    assert report["question_count"] == 2
    assert report["paired_deltas_candidate_minus_baseline"]["mrr"]["mean"] == 1.0
    assert output.exists()
