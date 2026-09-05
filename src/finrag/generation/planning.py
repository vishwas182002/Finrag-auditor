"""Answer planning, plan validation, and the auditable legacy heuristic."""

from __future__ import annotations

import ast
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from finrag.data.schemas import AnswerPlan, PlanValidation, RetrievalHit
from finrag.tools.calculator import UnsafeExpressionError, normalize_expression, safe_calculate
from finrag.tools.financial_numbers import parse_quantities

NUMBER_RE = re.compile(r"(?<![\w])[-]?\d[\d,]*(?:\.\d+)?")


def candidate_values(question: str, hits: list[RetrievalHit]) -> list[str]:
    """Replicate the original positional number scraper for baseline auditing."""
    question_years = {x for x in NUMBER_RE.findall(question) if len(x.replace(",", "")) == 4}
    values: list[str] = []
    for hit in hits[:3]:
        for raw in NUMBER_RE.findall(hit.chunk.content):
            cleaned = raw.replace(",", "")
            if cleaned in question_years:
                continue
            try:
                numeric = float(cleaned)
            except ValueError:
                continue
            if numeric.is_integer() and 1900 <= numeric <= 2100:
                continue
            if cleaned not in values:
                values.append(cleaned)
    return values


def legacy_plan_expression(question: str, hits: list[RetrievalHit]) -> str | None:
    """The pre-redesign keyword/position planner, retained only as a baseline."""
    lower = question.lower()
    values = candidate_values(question, hits)
    if len(values) < 2:
        return None
    if "average" in lower:
        chosen = values[: min(3, len(values))]
        return f"({' + '.join(chosen)}) / {len(chosen)}"
    if any(word in lower for word in ("percentage", "percent", "ratio")):
        if "change" in lower or "increase" in lower or "decrease" in lower:
            return f"(({values[0]}) - ({values[1]})) / ({values[1]}) * 100"
        return f"({values[0]}) / ({values[1]}) * 100"
    if any(word in lower for word in ("change", "difference", "increase", "decrease")):
        return f"({values[0]}) - ({values[1]})"
    if any(word in lower for word in ("total", "combined", "sum")):
        return f"({values[0]}) + ({values[1]})"
    return None


def legacy_answer_plan(question: str, hits: list[RetrievalHit]) -> AnswerPlan:
    """Wrap the old heuristic in the new typed interface for offline infrastructure tests."""
    if not hits:
        return AnswerPlan(
            decision="abstain",
            answer_type="none",
            reason_code="no_retrieved_evidence",
        )
    expression = legacy_plan_expression(question, hits)
    return AnswerPlan(
        decision="answer",
        answer_type="calculation" if expression else "extractive",
        selected_citation_ids=[hits[0].chunk.citation_id],
        calculator_expression=expression,
        result_unit="percent"
        if expression and any(word in question.lower() for word in ("percent", "ratio"))
        else "number",
        reason_code="deterministic_legacy_baseline",
    )


def _canonical_decimal(raw: str) -> str | None:
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def expression_operands(expression: str) -> list[str]:
    """Return canonical numeric literals from an already-safe arithmetic expression."""
    normalized = normalize_expression(expression)
    tree = ast.parse(normalized, mode="eval")
    operands: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            sign = "-" if isinstance(node.op, ast.USub) else ""
            canonical = _canonical_decimal(sign + str(node.operand.value))
            if canonical is not None:
                operands.append(canonical)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            canonical = _canonical_decimal(str(node.value))
            if canonical is not None:
                operands.append(canonical)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return operands


def evidence_operands(hits: list[RetrievalHit]) -> set[str]:
    values: set[str] = set()
    for hit in hits:
        for quantity in parse_quantities(hit.chunk.content):
            for value in (quantity.amount, quantity.value):
                canonical = _canonical_decimal(str(value))
                if canonical is not None:
                    values.add(canonical)
    return values


def validate_answer_plan(
    plan: AnswerPlan,
    retrieved: list[RetrievalHit],
    allowed_constants: tuple[str, ...],
    max_selected_chunks: int,
) -> tuple[PlanValidation, list[RetrievalHit]]:
    """Reject invented evidence, unsafe expressions, and unsupported numeric operands."""
    lookup = {hit.chunk.citation_id: hit for hit in retrieved}
    invalid_ids = [citation for citation in plan.selected_citation_ids if citation not in lookup]
    valid_ids = [citation for citation in plan.selected_citation_ids if citation in lookup]
    selected = [lookup[citation] for citation in valid_ids]
    reasons: list[str] = []
    unsupported: list[str] = []
    if plan.decision == "abstain":
        if plan.calculator_expression:
            reasons.append("abstention_plan_contains_expression")
        return (
            PlanValidation(
                valid=not reasons,
                valid_citation_ids=valid_ids,
                invalid_citation_ids=invalid_ids,
                unsupported_operands=[],
                reasons=reasons,
            ),
            selected,
        )
    if not plan.selected_citation_ids:
        reasons.append("answer_plan_has_no_evidence")
    if plan.answer_type == "none":
        reasons.append("answer_plan_has_no_answer_type")
    if len(set(plan.selected_citation_ids)) != len(plan.selected_citation_ids):
        reasons.append("duplicate_plan_citation")
    if invalid_ids:
        reasons.append("plan_citation_not_retrieved")
    if len(plan.selected_citation_ids) > max_selected_chunks:
        reasons.append("too_many_selected_evidence_chunks")
    if plan.answer_type == "calculation" and not plan.calculator_expression:
        reasons.append("calculation_plan_has_no_expression")
    if plan.answer_type != "calculation" and plan.calculator_expression:
        reasons.append("non_calculation_plan_contains_expression")
    if plan.calculator_expression:
        try:
            safe_calculate(plan.calculator_expression)
            expression_values = expression_operands(plan.calculator_expression)
        except (UnsafeExpressionError, SyntaxError, ValueError, ArithmeticError):
            reasons.append("unsafe_or_invalid_expression")
            expression_values = []
        available = evidence_operands(selected)
        constants = {
            canonical
            for raw in allowed_constants
            if (canonical := _canonical_decimal(raw)) is not None
        }
        unsupported = [
            value
            for value in expression_values
            if value not in available and value not in constants
        ]
        if unsupported:
            reasons.append("expression_operand_not_in_selected_evidence")
    return (
        PlanValidation(
            valid=not reasons,
            valid_citation_ids=valid_ids,
            invalid_citation_ids=invalid_ids,
            unsupported_operands=unsupported,
            reasons=reasons,
        ),
        selected,
    )


def counter_overlap(left: list[str], right: list[str]) -> int:
    """Multiset overlap used by the planner audit."""
    return sum((Counter(left) & Counter(right)).values())
