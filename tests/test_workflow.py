from __future__ import annotations

from typing import Any

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import AppConfig, EvidenceConfig
from finrag.data.schemas import AnswerPlan, RetrievalHit
from finrag.generation.providers import ExtractiveProvider, GenerationProvider
from finrag.tools.retrieval import evidence_sufficiency


class StubIndex:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.last_timing_ms = {"retrieval": 0.1}

    def search(self, *args: Any, **kwargs: Any) -> list[RetrievalHit]:
        return self.hits


def test_evidence_sufficiency_routing(hits) -> None:
    sufficient, details = evidence_sufficiency(
        "What was the change in revenue?", hits, EvidenceConfig(min_token_overlap=0.1)
    )
    assert sufficient is True
    assert details["decision"] == "answer"
    insufficient, _ = evidence_sufficiency(
        "Who is the chief scientist?", hits, EvidenceConfig(min_token_overlap=0.5)
    )
    assert insufficient is False


def test_langgraph_answer_state_transitions(hits) -> None:
    config = AppConfig(evidence=EvidenceConfig(min_token_overlap=0.1))
    workflow = FinRAGWorkflow(StubIndex(hits), ExtractiveProvider(), config)  # type: ignore[arg-type]
    result = workflow.answer("What was the change in revenue?")
    nodes = [event["node"] for event in result.trace]
    assert nodes == [
        "retrieve_evidence",
        "evidence_sufficiency",
        "plan_answer",
        "validate_plan",
        "calculate",
        "generate_answer",
        "verify_citations",
    ]
    assert result.abstained is False
    assert result.calculator_result == "20"
    assert result.plan_validation is not None
    assert result.plan_validation.valid is True
    assert result.citation_verification.valid is True


def test_langgraph_abstention_behavior() -> None:
    workflow = FinRAGWorkflow(StubIndex([]), ExtractiveProvider(), AppConfig())  # type: ignore[arg-type]
    result = workflow.answer("What is missing?")
    assert result.abstained is True
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert [event["node"] for event in result.trace][-1] == "abstain"


class UnsupportedOperandProvider(GenerationProvider):
    name = "test-unsupported-operand"

    def plan(self, question: str, hits: list[RetrievalHit]) -> AnswerPlan:
        return AnswerPlan(
            decision="answer",
            answer_type="calculation",
            selected_citation_ids=[hits[0].chunk.citation_id],
            calculator_expression="120 - 999",
            reason_code="test",
        )

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        plan: AnswerPlan,
        calculator_result: str | None = None,
    ) -> str:
        raise AssertionError("generation must not run for an invalid plan")


def test_plan_validation_blocks_unsupported_operand(hits) -> None:
    config = AppConfig(evidence=EvidenceConfig(min_token_overlap=0.1))
    workflow = FinRAGWorkflow(StubIndex(hits), UnsupportedOperandProvider(), config)  # type: ignore[arg-type]
    result = workflow.answer("What was the change in revenue?")
    assert result.abstained is True
    assert result.plan_validation is not None
    assert result.plan_validation.unsupported_operands == ["999"]
    assert "calculate" not in [event["node"] for event in result.trace]
