"""Regression cases for cross-report credit, units, grounding and resume identity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import AppConfig, EvidenceConfig
from finrag.data.schemas import AnswerPlan, AnswerResult, CitationVerification, RetrievalHit
from finrag.evaluation.answer_metrics import exact_match, numerical_match
from finrag.evaluation.citation_metrics import citation_metrics
from finrag.evaluation.provenance import digest, evaluation_identity, source_digest
from finrag.evaluation.retrieval_metrics import retrieval_metrics
from finrag.generation.planning import expression_operands, validate_answer_plan
from finrag.generation.providers import ExtractiveProvider
from finrag.tools.citations import render_calculation_answer, verify_answer
from finrag.tools.retrieval import evidence_sufficiency


def test_wrong_company_with_same_row_id_receives_no_credit(hits: list[RetrievalHit]) -> None:
    wrong = hits[0].model_copy(
        update={"chunk": hits[0].chunk.model_copy(update={"report_id": "OTHER/2023/report"})}
    )
    assert all(
        value == 0
        for value in retrieval_metrics([wrong], {"table_1"}, hits[0].chunk.report_id).values()
    )
    assert retrieval_metrics([wrong, hits[0]], {"table_1"}, hits[0].chunk.report_id)["mrr"] == 0.5
    result = AnswerResult(
        question="q",
        answer="120",
        abstained=False,
        citations=[wrong.chunk.citation_id],
        retrieved=[wrong],
        citation_verification=CitationVerification(
            valid=True,
            cited_ids=[wrong.chunk.citation_id],
            valid_ids=[wrong.chunk.citation_id],
            invalid_ids=[],
            malformed=[],
            reasons=[],
        ),
        trace=[],
        latency_ms={},
        provider="test",
    )
    scores = citation_metrics(result, {"table_1"}, hits[0].chunk.report_id)
    assert scores["citation_precision"] == scores["citation_recall"] == 0


def test_overlapping_chunks_do_not_repeat_ndcg_credit(hits: list[RetrievalHit]) -> None:
    duplicates = [hits[0].model_copy(update={"rank": rank}) for rank in range(1, 6)]
    assert retrieval_metrics(duplicates, {"table_1"}, hits[0].chunk.report_id)["ndcg@5"] == 1


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("2 million", "2 billion", False),
        ("2,000,000", "2 million", True),
        ("$2", "€2", False),
        ("GBP 2", "2 dollars", False),
        ("$2 million", "2000000 USD", True),
        ("2 billion", "2000 million", True),
        ("(2 million)", "-2000000", True),
        ("($2.5 million)", "-2500000 USD", True),
        ("−2 million", "-2000000", True),
        (".25", "0.25", True),
        ("2e6", "2 million", True),
        ("20 percent", "20%", True),
        ("0.2", "20%", False),
        ("0 million", "0 billion", True),
        ("$2 EUR", "$2 EUR", False),
    ],
)
def test_scale_sign_currency_and_percent(prediction: str, gold: str, expected: bool) -> None:
    assert exact_match(prediction, gold) is expected
    assert numerical_match(prediction, gold, 0.02) is expected


def test_calculated_answer_cannot_be_rewritten_by_provider(hits: list[RetrievalHit]) -> None:
    class Index:
        last_timing_ms: dict[str, float] = {}

        def search(self, *args: Any, **kwargs: Any) -> list[RetrievalHit]:
            return hits

    class FabricatingProvider(ExtractiveProvider):
        def generate(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("Calculation output must bypass model generation")

    workflow = FinRAGWorkflow(Index(), FabricatingProvider(), AppConfig())  # type: ignore[arg-type]
    result = workflow.answer("What was the change in revenue?")
    assert not result.abstained
    assert result.answer == f"20 [CITATION: {hits[0].chunk.citation_id}]"


def test_wrong_result_and_fabricated_prose_fail_verification(hits: list[RetrievalHit]) -> None:
    plan = AnswerPlan(
        decision="answer",
        answer_type="calculation",
        selected_citation_ids=[hits[0].chunk.citation_id],
        calculator_expression="120-100",
        reason_code="test",
    )
    correct = render_calculation_answer("20", plan)
    assert verify_answer(correct, hits, plan, "20").valid
    assert not verify_answer(correct.replace("20 [", "999 ["), hits, plan, "20").valid
    assert not verify_answer("Revenue doubled. " + correct, hits, plan, "20").valid
    assert not verify_answer(correct, hits, plan, None).valid


def test_extraction_rejects_invented_numbers_dates_and_relationships(
    hits: list[RetrievalHit],
) -> None:
    plan = AnswerPlan(
        decision="answer",
        answer_type="extractive",
        selected_citation_ids=[hits[0].chunk.citation_id],
        reason_code="test",
    )
    citation = f"[CITATION: {hits[0].chunk.citation_id}]"
    assert verify_answer(f"{hits[0].chunk.content} {citation}", hits, plan).valid
    for claim in (
        "Revenue was 999 in 2023.",
        "Revenue was 120 in 2021.",
        "Revenue doubled.",
        "20",
        "202",
        "",
    ):
        assert not verify_answer(f"{claim} {citation}", hits, plan).valid


def test_fallback_abstains_without_separate_threshold(hits: list[RetrievalHit]) -> None:
    reranked = [hits[0].model_copy(update={"method": "hybrid_rerank", "score": 0.9})]
    sufficient, details = evidence_sufficiency(
        "revenue", reranked, EvidenceConfig(min_reranker_score=0.6), "fallback:token-overlap"
    )
    assert not sufficient
    assert details["uncalibrated_fallback"] == "true"
    sufficient, _ = evidence_sufficiency(
        "revenue",
        reranked,
        EvidenceConfig(min_reranker_score=10, fallback_min_reranker_score=0.8),
        "fallback:token-overlap",
    )
    assert sufficient
    sufficient, _ = evidence_sufficiency(
        "revenue",
        reranked,
        EvidenceConfig(fallback_min_reranker_score=0.95),
        "fallback:token-overlap",
    )
    assert not sufficient


def test_negative_operands_preserve_sign_and_accounting_notation(hits: list[RetrievalHit]) -> None:
    assert expression_operands("-120 + 100") == ["-120", "100"]
    negative = hits[0].model_copy(
        update={"chunk": hits[0].chunk.model_copy(update={"content": "Loss ($120), revenue 100"})}
    )
    plan = AnswerPlan(
        decision="answer",
        answer_type="calculation",
        selected_citation_ids=[negative.chunk.citation_id],
        calculator_expression="-120 + 100",
        reason_code="test",
    )
    assert validate_answer_plan(plan, [negative], (), 3)[0].valid
    assert not validate_answer_plan(plan, hits, (), 3)[0].valid


def test_checkpoint_identity_changes_with_data_code_and_corpus(
    tmp_path: Path, example, chunks
) -> None:
    config = AppConfig()
    config.data.path = tmp_path / "data.json"
    config.data.path.write_text('{"value": 1}')
    original = evaluation_identity(config, [example], chunks, "provider", {})
    config.data.path.write_text('{"value": 2}')
    assert digest(evaluation_identity(config, [example], chunks, "provider", {})) != digest(
        original
    )
    config.data.path.write_text('{"value": 1}')
    modified = [chunks[0].model_copy(update={"content": "edited evidence"}), *chunks[1:]]
    assert digest(evaluation_identity(config, [example], modified, "provider", {})) != digest(
        original
    )
    config.generation.request_interval_seconds = 12
    config.evaluation.resume = False
    assert evaluation_identity(config, [example], chunks, "provider", {}) == original
    source = tmp_path / "package"
    source.mkdir()
    (source / "code.py").write_text("value = 1")
    before = source_digest(source)
    (source / "code.py").write_text("value = 2")
    assert source_digest(source) != before


def test_archived_regrades_reproduce_committed_outputs(tmp_path: Path) -> None:
    import json

    from finrag.evaluation.regrade import regrade_predictions

    root = Path(__file__).resolve().parents[1]
    for label, source in [
        ("historical_workflow", "results/predictions.jsonl"),
        ("dev_quick", "results/dev_quick/predictions.jsonl"),
    ]:
        output = tmp_path / label
        report = regrade_predictions(root / "artifacts/legacy_v1" / source, output)
        saved = root / "artifacts/regraded_v2" / label
        expected = json.loads((saved / "summary.json").read_text())
        # Python 3.12 uses improved float summation. Permit only roundoff,
        # while retaining exact source hashes, identities, counts and flags.
        for key in ("retrieval", "answer", "citations_on_answered"):
            assert report.pop(key) == pytest.approx(expected.pop(key), rel=0, abs=1e-12)
        assert report == expected
        actual_rows = list(map(json.loads, (output / "rows.jsonl").read_text().splitlines()))
        expected_rows = list(map(json.loads, (saved / "rows.jsonl").read_text().splitlines()))
        for actual, expected_row in zip(actual_rows, expected_rows, strict=True):
            for key in ("retrieval", "answer", "citations", "previous_answer"):
                assert actual.pop(key) == pytest.approx(expected_row.pop(key), rel=0, abs=1e-12)
            assert actual == expected_row


def test_regrade_rejects_conflicting_report_identity(tmp_path: Path) -> None:
    import json

    from finrag.evaluation.regrade import regrade_predictions

    root = Path(__file__).resolve().parents[1]
    row = json.loads(
        (root / "artifacts/legacy_v1/results/predictions.jsonl").read_text().splitlines()[0]
    )
    row["retrieved"][0]["report_id"] = "different_report"
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="conflicts"):
        regrade_predictions(source, tmp_path / "output")
