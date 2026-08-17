"""Development-only calibration of the transparent evidence-sufficiency gate."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from finrag.config import AppConfig
from finrag.data.finqa import load_finqa, select_configured, selected_corpus_examples
from finrag.data.schemas import FinQAExample, RetrievalHit
from finrag.indexing.chunking import build_chunks
from finrag.indexing.index import RetrievalIndex
from finrag.tools.retrieval import evidence_sufficiency

LOGGER = logging.getLogger(__name__)


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    def accepts(row: dict[str, Any]) -> bool:
        return bool(row["base_rule_passes"] and float(row["top_reranker_score"]) >= threshold)

    answerable_acceptance = sum(accepts(row) for row in answerable) / max(len(answerable), 1)
    unanswerable_abstention = sum(not accepts(row) for row in unanswerable) / max(
        len(unanswerable), 1
    )
    return {
        "threshold": threshold,
        "answerable_acceptance_rate": answerable_acceptance,
        "unanswerable_abstention_rate": unanswerable_abstention,
        "balanced_gate_accuracy": (answerable_acceptance + unanswerable_abstention) / 2,
    }


def select_threshold(
    rows: list[dict[str, Any]], min_answerable_acceptance: float = 0.0
) -> tuple[dict[str, float], list[dict[str, float]]]:
    scores = sorted({float(row["top_reranker_score"]) for row in rows})
    candidates = [0.0]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:], strict=False))
    if scores:
        candidates.append(scores[-1] + 1e-9)
    metrics = [_threshold_metrics(rows, threshold) for threshold in sorted(set(candidates))]
    eligible = [
        row
        for row in metrics
        if row["answerable_acceptance_rate"] >= min_answerable_acceptance
    ]
    if not eligible:
        raise ValueError("No threshold satisfies min_answerable_acceptance")
    selected = max(
        eligible,
        key=lambda row: (
            row["balanced_gate_accuracy"],
            row["unanswerable_abstention_rate"],
            row["answerable_acceptance_rate"],
            row["threshold"],
        ),
    )
    return selected, metrics


def _collect_candidates(
    examples: list[FinQAExample],
    index: RetrievalIndex,
    allowed_report_ids: list[list[str] | None],
    candidate_k: int,
    label: str,
) -> list[list[RetrievalHit]]:
    candidate_lists: list[list[RetrievalHit]] = []
    for position, (example, allowed) in enumerate(
        zip(examples, allowed_report_ids, strict=True), 1
    ):
        candidate_lists.append(
            index.search(
                example.question,
                method="hybrid",
                top_k=candidate_k,
                allowed_report_ids=allowed,
            )
        )
        if position % 50 == 0 or position == len(examples):
            LOGGER.info("Abstention calibration %s candidates: %d/%d", label, position, len(examples))
    return candidate_lists


def run_abstention_calibration(config: AppConfig, project_root: Path) -> dict[str, Any]:
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
    all_report_ids = sorted({chunk.report_id for chunk in chunks})
    answerable_allowed: list[list[str] | None] = [None] * len(selected)
    unanswerable_allowed: list[list[str] | None] = [
        [report_id for report_id in all_report_ids if report_id != example.report_id]
        for example in selected
    ]
    answerable_candidates = _collect_candidates(
        selected,
        index,
        answerable_allowed,
        config.retrieval.candidate_k,
        "answerable",
    )
    unanswerable_candidates = _collect_candidates(
        selected,
        index,
        unanswerable_allowed,
        config.retrieval.candidate_k,
        "source-withheld",
    )
    started = time.perf_counter()
    questions = [example.question for example in selected]
    answerable_hits = index.reranker.rerank_many(
        questions, answerable_candidates, config.retrieval.top_k
    )
    unanswerable_hits = index.reranker.rerank_many(
        questions, unanswerable_candidates, config.retrieval.top_k
    )
    reranking_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    no_score_config = config.evidence.model_copy(update={"min_reranker_score": -1e30})
    for answerable, hit_groups in ((True, answerable_hits), (False, unanswerable_hits)):
        for example, hits in zip(selected, hit_groups, strict=True):
            _, details = evidence_sufficiency(example.question, hits, no_score_config)
            rows.append(
                {
                    "question_id": example.question_id,
                    "answerable": answerable,
                    "base_rule_passes": details["decision"] == "answer",
                    "query_token_overlap": details["query_token_overlap"],
                    "top_reranker_score": hits[0].score if hits else float("-inf"),
                    "top_citation_id": hits[0].chunk.citation_id if hits else None,
                }
            )
    unconstrained_best, _ = select_threshold(rows)
    selected_threshold, sweep = select_threshold(
        rows, config.evaluation.min_answerable_acceptance
    )
    zero_threshold = _threshold_metrics(rows, 0.0)
    report = {
        "metadata": {
            "split": config.data.split,
            "answerable_questions": len(selected),
            "constructed_unanswerable_questions": len(selected),
            "construction": "Identical dev question with its source report removed from retrieval.",
            "corpus_scope": config.data.corpus_scope,
            "corpus_chunks": len(chunks),
            "backends": index.backends,
            "reranking_seconds": reranking_seconds,
            "generation_model_called": False,
        },
        "zero_threshold": zero_threshold,
        "selection_constraint": {
            "min_answerable_acceptance": config.evaluation.min_answerable_acceptance
        },
        "selected": selected_threshold,
        "unconstrained_best": unconstrained_best,
        "threshold_sweep": sweep,
    }
    output_dir = project_root / "artifacts" / "abstention_calibration" / config.evaluation.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "abstention_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "abstention_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "metadata": report["metadata"],
        "zero_threshold": zero_threshold,
        "selection_constraint": report["selection_constraint"],
        "selected": selected_threshold,
        "unconstrained_best": unconstrained_best,
    }
