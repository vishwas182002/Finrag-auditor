"""Financial-aware exact and normalized numerical answer metrics."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from finrag.tools.citations import CITATION_RE


def normalize_text(text: str) -> str:
    without_citations = CITATION_RE.sub("", text)
    lowered = without_citations.lower().strip()
    lowered = re.sub(r"\b(the|a|an)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9.%()-]+", " ", lowered)
    return " ".join(lowered.split())


def parse_financial_number(text: str) -> tuple[Decimal, bool] | None:
    cleaned = CITATION_RE.sub("", text).strip()
    matches = re.findall(r"\(?[-]?[$€£]?\d[\d,]*(?:\.\d+)?\)?%?", cleaned)
    if not matches:
        return None
    raw = matches[-1].strip()
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace(",", "")
    raw = re.sub(r"[$€£]", "", raw)
    is_percent = raw.endswith("%")
    raw = raw.rstrip("%")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    return (-value if negative else value, is_percent)


def exact_match(prediction: str, gold: str) -> bool:
    pred_num = parse_financial_number(prediction)
    gold_num = parse_financial_number(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    return normalize_text(prediction) == normalize_text(gold)


def numerical_match(prediction: str, gold: str, tolerance: float) -> bool | None:
    pred = parse_financial_number(prediction)
    target = parse_financial_number(gold)
    if pred is None or target is None:
        return None
    pred_value, pred_percent = pred
    gold_value, gold_percent = target
    if pred_percent != gold_percent:
        return False
    if gold_value == 0:
        return pred_value == 0
    relative_error = abs(pred_value - gold_value) / abs(gold_value)
    return relative_error <= Decimal(str(tolerance))


def score_answer(prediction: str, gold: str, tolerance: float) -> dict[str, float]:
    numeric = numerical_match(prediction, gold, tolerance)
    return {
        "exact_match": float(exact_match(prediction, gold)),
        "numerical_accuracy": float(bool(numeric)),
    }
