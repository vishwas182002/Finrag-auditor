"""Operational graph state. It intentionally contains no hidden reasoning text."""

from __future__ import annotations

from typing import Any, TypedDict

from finrag.data.schemas import AnswerPlan, CitationVerification, PlanValidation, RetrievalHit


class GraphState(TypedDict, total=False):
    question_id: str | None
    question: str
    retrieval_method: str
    allowed_report_ids: list[str] | None
    retrieved: list[RetrievalHit]
    sufficient: bool
    sufficiency_details: dict[str, Any]
    plan: AnswerPlan
    plan_validation: PlanValidation
    selected_evidence: list[RetrievalHit]
    calculator_expression: str | None
    calculator_result: str | None
    calculator_error: str | None
    answer: str
    abstained: bool
    citation_verification: CitationVerification
    trace: list[dict[str, Any]]
    latency_ms: dict[str, float]
