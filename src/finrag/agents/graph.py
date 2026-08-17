"""Controlled LangGraph workflow with validated provider-directed tool use."""

from __future__ import annotations

import time
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from finrag.agents.state import GraphState
from finrag.config import AppConfig
from finrag.data.schemas import AnswerResult, CitationVerification
from finrag.generation.planning import validate_answer_plan
from finrag.generation.providers import GenerationProvider
from finrag.indexing.index import RetrievalIndex
from finrag.tools.calculator import UnsafeExpressionError, safe_calculate
from finrag.tools.citations import parse_citations, verify_citations
from finrag.tools.retrieval import evidence_sufficiency, retrieve_evidence


def _trace(state: GraphState, node: str, **details: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, **details}]


class FinRAGWorkflow:
    def __init__(
        self, index: RetrievalIndex, provider: GenerationProvider, config: AppConfig
    ) -> None:
        self.index = index
        self.provider = provider
        self.config = config
        builder = StateGraph(GraphState)
        builder.add_node("retrieve_evidence", self._retrieve)
        builder.add_node("evidence_sufficiency", self._sufficiency)
        builder.add_node("plan_answer", self._plan)
        builder.add_node("validate_plan", self._validate_plan)
        builder.add_node("calculate", self._calculate)
        builder.add_node("generate_answer", self._generate)
        builder.add_node("verify_citations", self._verify)
        builder.add_node("abstain", self._abstain)
        builder.add_edge(START, "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "evidence_sufficiency")
        builder.add_conditional_edges(
            "evidence_sufficiency",
            lambda state: "plan" if state["sufficient"] else "abstain",
            {"plan": "plan_answer", "abstain": "abstain"},
        )
        builder.add_edge("plan_answer", "validate_plan")
        builder.add_conditional_edges(
            "validate_plan",
            self._route_validated_plan,
            {"calculate": "calculate", "generate": "generate_answer", "abstain": "abstain"},
        )
        builder.add_conditional_edges(
            "calculate",
            lambda state: "abstain" if state.get("calculator_error") else "generate",
            {"abstain": "abstain", "generate": "generate_answer"},
        )
        builder.add_edge("generate_answer", "verify_citations")
        builder.add_edge("verify_citations", END)
        builder.add_edge("abstain", END)
        self.graph = builder.compile()

    def _retrieve(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        hits = retrieve_evidence(
            self.index,
            state["question"],
            method=state.get("retrieval_method"),
            allowed_report_ids=state.get("allowed_report_ids"),
        )
        elapsed = (time.perf_counter() - started) * 1000
        timing = {
            **state.get("latency_ms", {}),
            **self.index.last_timing_ms,
            "retrieve_node": elapsed,
        }
        return {
            "retrieved": hits,
            "latency_ms": timing,
            "trace": _trace(
                state,
                "retrieve_evidence",
                chunk_ids=[hit.chunk.citation_id for hit in hits],
                scores=[round(hit.score, 6) for hit in hits],
                method=state.get("retrieval_method", self.config.retrieval.method),
            ),
        }

    def _sufficiency(self, state: GraphState) -> dict[str, Any]:
        sufficient, details = evidence_sufficiency(
            state["question"], state["retrieved"], self.config.evidence
        )
        return {
            "sufficient": sufficient,
            "sufficiency_details": details,
            "trace": _trace(state, "evidence_sufficiency", **details),
        }

    def _plan(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        plan = self.provider.plan(state["question"], state["retrieved"])
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "plan": plan,
            "calculator_expression": plan.calculator_expression,
            "latency_ms": {**state.get("latency_ms", {}), "planning": elapsed},
            "trace": _trace(
                state,
                "plan_answer",
                provider=self.provider.name,
                decision=plan.decision,
                answer_type=plan.answer_type,
                selected_citation_ids=plan.selected_citation_ids,
                calculator_expression=plan.calculator_expression,
                reason_code=plan.reason_code,
            ),
        }

    def _validate_plan(self, state: GraphState) -> dict[str, Any]:
        validation, selected = validate_answer_plan(
            state["plan"],
            state["retrieved"],
            self.config.planning.allowed_constants,
            self.config.planning.max_selected_chunks,
        )
        return {
            "plan_validation": validation,
            "selected_evidence": selected,
            "trace": _trace(
                state,
                "validate_plan",
                valid=validation.valid,
                reasons=validation.reasons,
                unsupported_operands=validation.unsupported_operands,
            ),
        }

    @staticmethod
    def _route_validated_plan(state: GraphState) -> str:
        if state["plan"].decision == "abstain" or not state["plan_validation"].valid:
            return "abstain"
        return "calculate" if state["plan"].answer_type == "calculation" else "generate"

    def _calculate(self, state: GraphState) -> dict[str, Any]:
        expression = state.get("calculator_expression")
        try:
            result = safe_calculate(expression or "")
            error = None
        except UnsafeExpressionError as exc:
            result = None
            error = str(exc)
        return {
            "calculator_result": result,
            "calculator_error": error,
            "trace": _trace(
                state, "calculate", expression=expression, result=result, error=error
            ),
        }

    def _generate(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = self.provider.generate(
            state["question"],
            state["selected_evidence"],
            state["plan"],
            state.get("calculator_result"),
        )
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "answer": answer,
            "abstained": answer.strip() == "INSUFFICIENT_EVIDENCE",
            "latency_ms": {**state.get("latency_ms", {}), "generation": elapsed},
            "trace": _trace(state, "generate_answer", provider=self.provider.name),
        }

    def _verify(self, state: GraphState) -> dict[str, Any]:
        verification = verify_citations(state["answer"], state["selected_evidence"])
        answer = state["answer"]
        abstained = state.get("abstained", False)
        if not verification.valid:
            answer = "INSUFFICIENT_EVIDENCE"
            abstained = True
        return {
            "answer": answer,
            "abstained": abstained,
            "citation_verification": verification,
            "trace": _trace(
                state,
                "verify_citations",
                valid=verification.valid,
                reasons=verification.reasons,
            ),
        }

    def _abstain(self, state: GraphState) -> dict[str, Any]:
        plan = state.get("plan")
        if not state.get("sufficient", False):
            reason = "evidence_below_configured_threshold"
        elif state.get("calculator_error"):
            reason = "calculator_rejected_expression"
        elif plan is not None and plan.decision == "abstain":
            reason = plan.reason_code
        else:
            reason = "plan_validation_failed"
        verification = CitationVerification(
            valid=True,
            cited_ids=[],
            valid_ids=[],
            invalid_ids=[],
            malformed=[],
            reasons=["abstained_before_generation"],
        )
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "abstained": True,
            "citation_verification": verification,
            "trace": _trace(state, "abstain", reason=reason),
        }

    def answer(
        self,
        question: str,
        question_id: str | None = None,
        retrieval_method: str | None = None,
        allowed_report_ids: list[str] | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        result = cast(
            GraphState,
            self.graph.invoke(
                {
                    "question_id": question_id,
                    "question": question,
                    "retrieval_method": retrieval_method or self.config.retrieval.method,
                    "allowed_report_ids": allowed_report_ids,
                    "trace": [],
                    "latency_ms": {},
                }
            ),
        )
        latency = dict(result.get("latency_ms", {}))
        latency["total"] = (time.perf_counter() - started) * 1000
        answer = result["answer"]
        return AnswerResult(
            question_id=question_id,
            question=question,
            answer=answer,
            abstained=result["abstained"],
            citations=parse_citations(answer),
            retrieved=result.get("retrieved", []),
            citation_verification=result["citation_verification"],
            plan=result.get("plan"),
            plan_validation=result.get("plan_validation"),
            calculator_expression=result.get("calculator_expression"),
            calculator_result=result.get("calculator_result"),
            trace=result["trace"],
            latency_ms=latency,
            provider=self.provider.name,
        )
