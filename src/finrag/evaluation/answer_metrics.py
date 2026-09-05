"""Versioned, scale-aware financial answer metrics."""

from __future__ import annotations

import re
from decimal import Decimal

from finrag.tools.citations import CITATION_RE
from finrag.tools.financial_numbers import answer_quantity, compatible_units, parse_quantities

ANSWER_METRIC_VERSION = "financial-units-v2"


def normalize_text(text: str) -> str:
    lowered = CITATION_RE.sub("", text).lower().strip()
    lowered = re.sub(r"\b(the|a|an)\b", " ", lowered)
    return " ".join(re.sub(r"[^a-z0-9.%()-]+", " ", lowered).split())


def parse_financial_numbers(text: str) -> list[tuple[Decimal, bool]]:
    """Compatibility interface; scoring additionally checks explicit currencies."""
    return [(quantity.value, quantity.is_percent) for quantity in parse_quantities(text)]


def parse_financial_number(text: str) -> tuple[Decimal, bool] | None:
    quantity = answer_quantity(text)
    return (quantity.value, quantity.is_percent) if quantity else None


def exact_match(prediction: str, gold: str) -> bool:
    pred, target = answer_quantity(prediction), answer_quantity(gold)
    if pred is not None and target is not None:
        return compatible_units(pred, target) and pred.value == target.value
    return normalize_text(prediction) == normalize_text(gold)


def numerical_match(prediction: str, gold: str, tolerance: float) -> bool | None:
    pred, target = answer_quantity(prediction), answer_quantity(gold)
    if pred is None or target is None:
        return None
    if not compatible_units(pred, target):
        return False
    if target.value == 0:
        return pred.value == 0
    return abs(pred.value - target.value) / abs(target.value) <= Decimal(str(tolerance))


def score_answer(prediction: str, gold: str, tolerance: float) -> dict[str, float]:
    return {
        "exact_match": float(exact_match(prediction, gold)),
        "numerical_accuracy": float(bool(numerical_match(prediction, gold, tolerance))),
    }
