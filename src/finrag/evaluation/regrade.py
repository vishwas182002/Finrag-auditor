"""Regrade saved workflow hits/answers without inference or changing raw artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from finrag.data.schemas import AnswerResult, CitationVerification, DocumentChunk, RetrievalHit
from finrag.evaluation.answer_metrics import ANSWER_METRIC_VERSION, score_answer
from finrag.evaluation.citation_metrics import citation_metrics
from finrag.evaluation.retrieval_metrics import METRIC_VERSION, mean_metrics, retrieval_metrics


def regrade_predictions(source: Path, output: Path, tolerance: float = 0.02) -> dict[str, Any]:
    payload = source.read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    corrected = []
    for row in rows:
        if not row["answerable"]:
            continue
        hits = []
        for saved in row["retrieved"]:
            report_id, chunk_id = saved["citation_id"].rsplit("/", 1)
            if saved.get("report_id", report_id) != report_id:
                raise ValueError("Saved report_id conflicts with citation_id")
            hits.append(
                RetrievalHit(
                    chunk=DocumentChunk(
                        report_id=report_id,
                        chunk_id=chunk_id,
                        source_section="table" if saved["source_type"] == "table" else "pre_text",
                        source_type=saved["source_type"],
                        source_ids=tuple(saved["source_ids"]),
                        content="",
                        raw_content="",
                    ),
                    score=saved["score"],
                    rank=saved["rank"],
                    method=saved["method"],
                )
            )
        result = AnswerResult(
            question=row["question"],
            answer=row["prediction"],
            abstained=row["abstained"],
            citations=row["citations"],
            retrieved=hits,
            citation_verification=CitationVerification.model_validate(row["citation_verification"]),
            trace=[],
            latency_ms={},
            provider=row["provider"],
        )
        corrected.append(
            {
                "evaluation_id": row["evaluation_id"],
                "question_id": row["question_id"],
                "report_id": row["report_id"],
                "abstained": row["abstained"],
                "retrieval": retrieval_metrics(hits, row["gold_source_ids"], row["report_id"]),
                "answer": score_answer(row["prediction"], row["gold_answer"], tolerance),
                "citations": citation_metrics(
                    result, set(row["gold_source_ids"]), row["report_id"]
                ),
                "previous_answer": row["answer_metrics"],
            }
        )
    answered = [row for row in corrected if not row["abstained"]]
    report = {
        "retrieval_metric_version": METRIC_VERSION,
        "answer_metric_version": ANSWER_METRIC_VERSION,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "kind": "saved_workflow_regrade",
        "fresh_inference": False,
        "new_answer_verification_applied": False,
        "limitations": "Only saved workflow hits can be regraded. Other retrieval methods lack saved hit identities. Original answer decisions and citation-reference checks are preserved; this is not a run of the new workflow.",
        "answerable_questions": len(corrected),
        "answered_questions": len(answered),
        "changed_answer_scores": sum(row["answer"] != row["previous_answer"] for row in corrected),
        "numerical_tolerance": tolerance,
        "retrieval": mean_metrics([row["retrieval"] for row in corrected]),
        "answer": mean_metrics([row["answer"] for row in corrected]),
        "citations_on_answered": mean_metrics([row["citations"] for row in answered]),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corrected)
    )
    return report
