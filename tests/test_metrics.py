from __future__ import annotations

from finrag.data.schemas import AnswerResult, CitationVerification, RetrievalHit
from finrag.evaluation.answer_metrics import exact_match, numerical_match, parse_financial_number
from finrag.evaluation.citation_metrics import citation_metrics
from finrag.evaluation.retrieval_metrics import retrieval_metrics


def test_financial_number_parsing() -> None:
    assert parse_financial_number("The answer is ($1,250.00).") == (-1250, False)
    assert parse_financial_number("15.7%") == (parse_financial_number("15.7%")[0], True)


def test_numerical_answer_comparison() -> None:
    assert numerical_match("The result is $102", "100", tolerance=0.02) is True
    assert numerical_match("0.15", "15%", tolerance=0.02) is False
    assert exact_match("1,200.0", "$1200") is True


def test_generated_percentage_with_unicode_space_is_recognized() -> None:
    prediction = "The calculated result is **35.099177552\u202f%** of the total."
    assert numerical_match(prediction, "35.1%", tolerance=0.02) is True


def test_percentage_answer_is_preferred_over_trailing_years() -> None:
    prediction = "The increase was **5.27\u202f%** from December 31 2017 to December 31 2018."
    assert numerical_match(prediction, "5.3%", tolerance=0.02) is True


def test_retrieval_metrics(hits: list[RetrievalHit]) -> None:
    metrics = retrieval_metrics(hits, {"table_1"}, hits[0].chunk.report_id)
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg@5"] == 1.0


def test_citation_metrics(hits: list[RetrievalHit]) -> None:
    citation = hits[0].chunk.citation_id
    result = AnswerResult(
        question="q",
        answer=f"Revenue was 120. [CITATION: {citation}]",
        abstained=False,
        citations=[citation],
        retrieved=hits,
        citation_verification=CitationVerification(
            valid=True,
            cited_ids=[citation],
            valid_ids=[citation],
            invalid_ids=[],
            malformed=[],
            reasons=[],
        ),
        trace=[],
        latency_ms={},
        provider="test",
    )
    metrics = citation_metrics(result, {"table_1"}, hits[0].chunk.report_id)
    assert metrics["citation_reference_integrity"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
