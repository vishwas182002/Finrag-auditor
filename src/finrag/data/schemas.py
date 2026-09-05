"""Shared Pydantic schemas used across ingestion, retrieval, and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class FinQAExample(FrozenModel):
    question_id: str
    report_id: str
    question: str
    answer: str
    program: str
    gold_source_ids: tuple[str, ...]
    pre_text: tuple[str, ...]
    post_text: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]


class DocumentChunk(FrozenModel):
    report_id: str
    chunk_id: str
    source_section: Literal["pre_text", "post_text", "table"]
    source_type: Literal["text", "table"]
    source_ids: tuple[str, ...]
    content: str
    raw_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_id(self) -> str:
        return f"{self.report_id}/{self.chunk_id}"

    @property
    def global_id(self) -> str:
        return f"{self.report_id}::{self.chunk_id}"


class RetrievalHit(FrozenModel):
    chunk: DocumentChunk
    score: float
    rank: int
    method: str
    component_scores: dict[str, float] = Field(default_factory=dict)


class CitationVerification(BaseModel):
    valid: bool
    cited_ids: list[str]
    valid_ids: list[str]
    invalid_ids: list[str]
    malformed: list[str]
    reasons: list[str]


class AnswerPlan(BaseModel):
    """Operational plan returned by a provider; contains no free-form reasoning."""

    decision: Literal["answer", "abstain"]
    answer_type: Literal["calculation", "extractive", "none"]
    selected_citation_ids: list[str] = Field(default_factory=list)
    calculator_expression: str | None = None
    # Expressions must yield base units, or percentage points for percent.
    result_unit: Literal["number", "percent", "USD", "EUR", "GBP"] = "number"
    reason_code: str


class PlanValidation(BaseModel):
    valid: bool
    valid_citation_ids: list[str]
    invalid_citation_ids: list[str]
    unsupported_operands: list[str]
    reasons: list[str]


class AnswerResult(BaseModel):
    question_id: str | None = None
    question: str
    answer: str
    abstained: bool
    citations: list[str]
    retrieved: list[RetrievalHit]
    citation_verification: CitationVerification
    plan: AnswerPlan | None = None
    plan_validation: PlanValidation | None = None
    calculator_expression: str | None = None
    calculator_result: str | None = None
    trace: list[dict[str, Any]]
    latency_ms: dict[str, float]
    provider: str
