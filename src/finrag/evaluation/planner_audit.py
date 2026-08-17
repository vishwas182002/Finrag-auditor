"""Isolated audit of the legacy arithmetic planner against FinQA gold programs."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from finrag.config import AppConfig
from finrag.data.finqa import load_finqa, select_configured, selected_corpus_examples
from finrag.data.schemas import FinQAExample, RetrievalHit
from finrag.evaluation.answer_metrics import parse_financial_number
from finrag.generation.planning import expression_operands, legacy_plan_expression
from finrag.indexing.chunking import build_chunks, chunk_report
from finrag.indexing.index import RetrievalIndex
from finrag.tools.calculator import UnsafeExpressionError, normalize_expression, safe_calculate

PROGRAM_CALL_RE = re.compile(r"([a-z_]+)\(([^()]*)\)")
REFERENCE_RE = re.compile(r"^#\d+$")
OPERATOR_TYPES = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.Pow: "exp",
}


def _canonical_number(raw: str) -> str | None:
    cleaned = raw.strip()
    if cleaned == "const_m1":
        cleaned = "-1"
    elif cleaned.startswith("const_"):
        cleaned = cleaned.removeprefix("const_")
    try:
        value = Decimal(cleaned.replace(",", ""))
    except Exception:
        return None
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def parse_gold_program(program: str) -> tuple[list[str], list[str], list[str]]:
    """Return operation sequence, evidence operands, and explicit constants."""
    operators: list[str] = []
    operands: list[str] = []
    constants: list[str] = []
    for operator, arguments in PROGRAM_CALL_RE.findall(program):
        operators.append(operator)
        for argument in (part.strip() for part in arguments.split(",")):
            if REFERENCE_RE.match(argument):
                continue
            canonical = _canonical_number(argument)
            if canonical is None:
                continue
            if argument.startswith("const_"):
                constants.append(canonical)
            else:
                operands.append(canonical)
    return operators, operands, constants


def expression_operators(expression: str) -> list[str]:
    tree = ast.parse(normalize_expression(expression), mode="eval")
    operators: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.BinOp):
            visit(node.left)
            visit(node.right)
            operators.append(OPERATOR_TYPES.get(type(node.op), type(node.op).__name__.lower()))
        elif isinstance(node, ast.UnaryOp):
            visit(node.operand)

    visit(tree.body)
    return operators


def _remove_gold_constants(predicted: list[str], constants: list[str]) -> list[str]:
    remaining = Counter(predicted)
    for constant in constants:
        if remaining[constant] > 0:
            remaining[constant] -= 1
    return sorted(remaining.elements())


def _execution_matches(expression: str | None, gold_answer: str, tolerance: float) -> bool:
    if expression is None:
        return False
    try:
        predicted = Decimal(safe_calculate(expression))
    except (UnsafeExpressionError, ArithmeticError):
        return False
    parsed_gold = parse_financial_number(gold_answer)
    if parsed_gold is None:
        return False
    gold, _ = parsed_gold
    if gold == 0:
        return predicted == 0
    return abs(predicted - gold) / abs(gold) <= Decimal(str(tolerance))


def score_legacy_plan(
    example: FinQAExample, hits: list[RetrievalHit], tolerance: float
) -> dict[str, Any]:
    expression = legacy_plan_expression(example.question, hits)
    gold_operators, gold_operands, gold_constants = parse_gold_program(example.program)
    predicted_operators: list[str] = []
    predicted_operands: list[str] = []
    executable = False
    if expression is not None:
        try:
            safe_calculate(expression)
            predicted_operators = expression_operators(expression)
            predicted_operands = _remove_gold_constants(
                expression_operands(expression), gold_constants
            )
            executable = True
        except (UnsafeExpressionError, SyntaxError, ValueError):
            pass
    operand_overlap = sum((Counter(predicted_operands) & Counter(gold_operands)).values())
    operand_accuracy = bool(gold_operands) and Counter(predicted_operands) == Counter(gold_operands)
    operator_accuracy = bool(gold_operators) and predicted_operators == gold_operators
    return {
        "expression": expression,
        "gold_program": example.program,
        "gold_operators": gold_operators,
        "predicted_operators": predicted_operators,
        "gold_operands": gold_operands,
        "predicted_operands": predicted_operands,
        "planner_attempted": float(expression is not None),
        "executable": float(executable),
        "operator_accuracy": float(operator_accuracy),
        "operand_multiset_accuracy": float(operand_accuracy),
        "operand_recall": operand_overlap / max(len(gold_operands), 1),
        "program_structure_accuracy": float(operator_accuracy and operand_accuracy),
        "execution_accuracy": float(
            _execution_matches(expression, example.answer, tolerance)
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "gold_evidence_available",
        "planner_attempted",
        "executable",
        "operator_accuracy",
        "operand_multiset_accuracy",
        "operand_recall",
        "program_structure_accuracy",
        "execution_accuracy",
    ]
    return {
        metric: sum(float(row[metric]) for row in rows) / max(len(rows), 1)
        for metric in metric_names
    }


def _oracle_hits(example: FinQAExample, config: AppConfig) -> list[RetrievalHit]:
    chunks = chunk_report(example, config.chunking)
    gold = set(example.gold_source_ids)
    selected = [chunk for chunk in chunks if set(chunk.source_ids) & gold]
    return [
        RetrievalHit(chunk=chunk, score=1.0, rank=rank, method="oracle_gold_evidence")
        for rank, chunk in enumerate(selected, 1)
    ]


def run_planner_audit(config: AppConfig, project_root: Path) -> dict[str, Any]:
    """Audit current retrieval and oracle evidence without calling a generation model."""
    examples = load_finqa(config.data.path)
    selected = select_configured(
        examples,
        config.data.sample_size,
        config.seed,
        config.data.question_manifest,
    )
    corpus = selected_corpus_examples(examples, selected, config.data.corpus_scope)
    index = RetrievalIndex(build_chunks(corpus, config.chunking), config.retrieval)
    retrieved_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for example in selected:
        retrieved = index.search(
            example.question,
            method=config.retrieval.method,
            top_k=config.retrieval.top_k,
        )
        gold = set(example.gold_source_ids)
        evidence_available = any(set(hit.chunk.source_ids) & gold for hit in retrieved)
        base = {
            "question_id": example.question_id,
            "report_id": example.report_id,
            "question": example.question,
            "gold_answer": example.answer,
        }
        retrieved_rows.append(
            {
                **base,
                "condition": "retrieved",
                "gold_evidence_available": float(evidence_available),
                "retrieved_citation_ids": [hit.chunk.citation_id for hit in retrieved],
                **score_legacy_plan(example, retrieved, config.evaluation.numerical_tolerance),
            }
        )
        oracle = _oracle_hits(example, config)
        oracle_rows.append(
            {
                **base,
                "condition": "oracle",
                "gold_evidence_available": float(bool(oracle)),
                "retrieved_citation_ids": [hit.chunk.citation_id for hit in oracle],
                **score_legacy_plan(example, oracle, config.evaluation.numerical_tolerance),
            }
        )
    available_rows = [row for row in retrieved_rows if row["gold_evidence_available"]]
    report = {
        "metadata": {
            "split": config.data.split,
            "sample_size": len(selected),
            "retrieval_method": config.retrieval.method,
            "retrieval_backends": index.backends,
            "gold_program_used_only_for_scoring": True,
            "generation_model_called": False,
        },
        "retrieved_evidence": _aggregate(retrieved_rows),
        "retrieved_evidence_conditional_on_gold_available": _aggregate(available_rows),
        "oracle_gold_evidence": _aggregate(oracle_rows),
    }
    output_dir = project_root / "artifacts" / "planner_audit" / config.evaluation.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "planner_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "planner_audit_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in [*retrieved_rows, *oracle_rows]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return report
