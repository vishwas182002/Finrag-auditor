from __future__ import annotations

from finrag.evaluation.planner_audit import parse_gold_program, score_legacy_plan


def test_gold_program_parsing_preserves_operations_and_operands() -> None:
    operators, operands, constants = parse_gold_program(
        "subtract(120, 100), divide(#0, 100), multiply(#1, const_100)"
    )
    assert operators == ["subtract", "divide", "multiply"]
    assert operands == ["120", "100", "100"]
    assert constants == ["100"]


def test_legacy_planner_scores_against_gold(example, hits) -> None:
    scores = score_legacy_plan(example, hits, tolerance=0.02)
    assert scores["operator_accuracy"] == 1.0
    assert scores["operand_multiset_accuracy"] == 1.0
    assert scores["execution_accuracy"] == 1.0
