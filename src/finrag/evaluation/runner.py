"""End-to-end held-out FinQA ablation and agent workflow evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import AppConfig
from finrag.data.finqa import load_finqa, select_configured, selected_corpus_examples
from finrag.data.schemas import AnswerResult, FinQAExample
from finrag.evaluation.answer_metrics import score_answer
from finrag.evaluation.bootstrap import bootstrap_mean_ci
from finrag.evaluation.checkpoint import JsonlCheckpoint
from finrag.evaluation.citation_metrics import citation_metrics
from finrag.evaluation.latency import latency_summary
from finrag.evaluation.retrieval_metrics import mean_metrics, retrieval_metrics
from finrag.generation.providers import create_provider
from finrag.indexing.chunking import build_chunks
from finrag.indexing.index import RetrievalIndex

LOGGER = logging.getLogger(__name__)


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return mean_metrics(rows) if rows else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _portable_path(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _prediction_row(
    evaluation_id: str,
    example: FinQAExample,
    result: AnswerResult,
    answerable: bool,
    tolerance: float,
) -> dict[str, Any]:
    answer_scores = score_answer(result.answer, example.answer, tolerance) if answerable else {}
    citation_scores = (
        citation_metrics(result, set(example.gold_source_ids)) if answerable else {}
    )
    return {
        "evaluation_id": evaluation_id,
        "question_id": example.question_id,
        "report_id": example.report_id,
        "question": example.question,
        "gold_answer": example.answer if answerable else None,
        "gold_source_ids": list(example.gold_source_ids) if answerable else [],
        "answerable": answerable,
        "prediction": result.answer,
        "abstained": result.abstained,
        "citations": result.citations,
        "citation_verification": result.citation_verification.model_dump(),
        "plan": result.plan.model_dump() if result.plan else None,
        "plan_validation": (
            result.plan_validation.model_dump() if result.plan_validation else None
        ),
        "answer_metrics": answer_scores,
        "citation_metrics": citation_scores,
        "calculator_expression": result.calculator_expression,
        "calculator_result": result.calculator_result,
        "retrieved": [
            {
                "citation_id": hit.chunk.citation_id,
                "source_ids": list(hit.chunk.source_ids),
                "source_type": hit.chunk.source_type,
                "score": hit.score,
                "rank": hit.rank,
                "method": hit.method,
                "component_scores": hit.component_scores,
            }
            for hit in result.retrieved
        ],
        "trace": result.trace,
        "latency_ms": result.latency_ms,
        "provider": result.provider,
    }


def run_evaluation(config: AppConfig, project_root: Path) -> dict[str, Any]:
    started = time.time()
    results_dir = project_root / "artifacts" / "results"
    if config.evaluation.run_name != "default":
        results_dir = results_dir / config.evaluation.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    examples = load_finqa(config.data.path)
    selected = select_configured(
        examples,
        config.data.sample_size,
        config.seed,
        config.data.question_manifest,
    )
    corpus_examples = selected_corpus_examples(
        examples, selected, config.data.corpus_scope
    )
    chunks = build_chunks(corpus_examples, config.chunking)
    LOGGER.info("Building indexes for %d chunks across %d reports", len(chunks), len({c.report_id for c in chunks}))
    index = RetrievalIndex(chunks, config.retrieval, allow_model_fallback=True)

    methods = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    retrieval_report: dict[str, Any] = {}
    ablation_rows: list[dict[str, Any]] = []
    retrieval_detail: list[dict[str, Any]] = []
    for method in methods:
        per_query: list[dict[str, float]] = []
        retrieval_latencies: list[float] = []
        for example in selected:
            query_started = time.perf_counter()
            hits = index.search(example.question, method=method, top_k=5)
            retrieval_latencies.append((time.perf_counter() - query_started) * 1000)
            metrics = retrieval_metrics(hits, example.gold_source_ids)
            per_query.append(metrics)
            retrieval_detail.append(
                {"method": method, "question_id": example.question_id, **metrics}
            )
        aggregate = _mean(per_query)
        cis = {
            key: bootstrap_mean_ci(
                [row[key] for row in per_query], config.evaluation.bootstrap_samples, config.seed
            )
            for key in aggregate
        }
        retrieval_report[method] = {
            "metrics": aggregate,
            "bootstrap_95_ci": cis,
            "median_latency_ms": float(np.median(retrieval_latencies)),
            "p95_latency_ms": float(np.percentile(retrieval_latencies, 95)),
        }
        ablation_rows.append(
            {
                "configuration": method,
                **aggregate,
                "answer_accuracy": "",
                "citation_reference_integrity": "",
                "coverage": "",
                "median_latency_ms": float(np.median(retrieval_latencies)),
            }
        )

    provider = create_provider(
        config.generation.provider,
        config.generation.model,
        config.generation.temperature,
        config.generation.base_url,
        config.generation.api_key_env,
        config.generation.max_retries,
    )
    workflow = FinRAGWorkflow(index, provider, config)
    fingerprint_config = config.model_dump(mode="json")
    fingerprint_config["data"]["path"] = _portable_path(config.data.path, project_root)
    fingerprint_config["data"]["question_manifest"] = _portable_path(
        config.data.question_manifest, project_root
    )
    fingerprint_payload = {
        "config": fingerprint_config,
        "question_ids": [example.question_id for example in selected],
        "provider": provider.name,
        "backends": index.backends,
    }
    evaluation_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    checkpoint = JsonlCheckpoint(
        project_root
        / "artifacts"
        / "checkpoints"
        / f"workflow_{config.evaluation.run_name}_{evaluation_fingerprint}.jsonl"
    )
    existing_rows = checkpoint.rows() if config.evaluation.resume else []
    existing = {str(row["evaluation_id"]): row for row in existing_rows}
    prediction_rows: list[dict[str, Any]] = []
    all_report_ids = sorted({chunk.report_id for chunk in chunks})

    evaluation_cases: list[tuple[str, FinQAExample, bool, list[str] | None]] = []
    for example in selected:
        evaluation_cases.append((f"answerable:{example.question_id}", example, True, None))
    for example in selected[: config.data.unanswerable_size]:
        wrong_corpus = [report for report in all_report_ids if report != example.report_id]
        evaluation_cases.append((f"unanswerable:{example.question_id}", example, False, wrong_corpus))

    for evaluation_id, example, answerable, allowed_reports in evaluation_cases:
        if evaluation_id in existing:
            row = existing[evaluation_id]
        else:
            result = workflow.answer(
                question=example.question,
                question_id=example.question_id,
                retrieval_method="hybrid_rerank",
                allowed_report_ids=allowed_reports,
            )
            row = _prediction_row(
                evaluation_id,
                example,
                result,
                answerable,
                config.evaluation.numerical_tolerance,
            )
            checkpoint.append(row)
        prediction_rows.append(row)

    answerable_rows = [row for row in prediction_rows if row["answerable"]]
    unanswerable_rows = [row for row in prediction_rows if not row["answerable"]]
    answered_rows = [row for row in answerable_rows if not row["abstained"]]
    answer_metric_rows = [row["answer_metrics"] for row in answerable_rows]
    citation_metric_rows = [row["citation_metrics"] for row in answered_rows]
    answer_aggregate = _mean(answer_metric_rows)
    citation_aggregate = _mean(citation_metric_rows)
    answer_report = {
        "provider": provider.name,
        "run_name": config.evaluation.run_name,
        "evaluation_fingerprint": evaluation_fingerprint,
        "question_manifest": _portable_path(config.data.question_manifest, project_root),
        "question_ids_sha256": hashlib.sha256(
            ("\n".join(example.question_id for example in selected) + "\n").encode()
        ).hexdigest(),
        "answerable_questions": len(answerable_rows),
        "answered_questions": len(answered_rows),
        "coverage": len(answered_rows) / max(len(answerable_rows), 1),
        "abstention_rate": sum(row["abstained"] for row in answerable_rows)
        / max(len(answerable_rows), 1),
        "accuracy_on_answered": (
            sum(row["answer_metrics"]["numerical_accuracy"] for row in answered_rows)
            / max(len(answered_rows), 1)
        ),
        **answer_aggregate,
        "correct_unanswerable_abstention_rate": (
            sum(row["abstained"] for row in unanswerable_rows)
            / max(len(unanswerable_rows), 1)
        ),
        "constructed_unanswerable_questions": len(unanswerable_rows),
        "bootstrap_95_ci": {
            key: bootstrap_mean_ci(
                [row[key] for row in answer_metric_rows],
                config.evaluation.bootstrap_samples,
                config.seed,
            )
            for key in answer_aggregate
        },
    }
    citation_report = {
        "evaluated_answered_questions": len(answered_rows),
        **citation_aggregate,
    }
    workflow_latencies = [row["latency_ms"] for row in prediction_rows]
    latency_report = latency_summary(workflow_latencies)
    metadata = {
        "created_unix": started,
        "duration_seconds": time.time() - started,
        "python": platform.python_version(),
        "split": config.data.split,
        "split_path": _portable_path(config.data.path, project_root),
        "sample_size": len(selected),
        "report_count": len({example.report_id for example in selected}),
        "corpus_scope": config.data.corpus_scope,
        "corpus_chunks": len(chunks),
        "text_supported_questions": sum(
            any(source.startswith("text_") for source in example.gold_source_ids)
            for example in selected
        ),
        "table_supported_questions": sum(
            any(source.startswith("table_") for source in example.gold_source_ids)
            for example in selected
        ),
        "chunking": config.chunking.model_dump(),
        "retrieval_backends": index.backends,
        "provider": provider.name,
        "run_name": config.evaluation.run_name,
        "evaluation_fingerprint": evaluation_fingerprint,
        "question_manifest": _portable_path(config.data.question_manifest, project_root),
        "question_ids_sha256": hashlib.sha256(
            ("\n".join(example.question_id for example in selected) + "\n").encode()
        ).hexdigest(),
        "seed": config.seed,
        "gold_evidence_used_for_retrieval": False,
    }

    ablation_rows.append(
        {
            "configuration": "full_workflow",
            **retrieval_report["hybrid_rerank"]["metrics"],
            "answer_accuracy": answer_report["numerical_accuracy"],
            "citation_reference_integrity": citation_report.get(
                "citation_reference_integrity", 0.0
            ),
            "coverage": answer_report["coverage"],
            "median_latency_ms": latency_report.get("total", {}).get("median_ms", 0.0),
        }
    )

    (results_dir / "retrieval_metrics.json").write_text(
        json.dumps({"metadata": metadata, "methods": retrieval_report}, indent=2, default=_json_default)
        + "\n"
    )
    (results_dir / "answer_metrics.json").write_text(
        json.dumps({"metadata": metadata, "answer": answer_report, "citations": citation_report}, indent=2)
        + "\n"
    )
    with (results_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    latency_keys = ["evaluation_id", "question_id", "answerable", "retrieval", "reranking", "generation", "total"]
    with (results_dir / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=latency_keys)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow(
                {
                    "evaluation_id": row["evaluation_id"],
                    "question_id": row["question_id"],
                    "answerable": row["answerable"],
                    **{key: row["latency_ms"].get(key, "") for key in latency_keys[3:]},
                }
            )
    with (results_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    (results_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "metadata": metadata,
        "retrieval": retrieval_report,
        "answer": answer_report,
        "citations": citation_report,
        "latency": latency_report,
        "ablation": ablation_rows,
    }
