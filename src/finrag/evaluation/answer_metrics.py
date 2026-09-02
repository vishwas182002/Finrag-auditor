"""Financial-aware exact and normalized numerical answer metrics."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from finrag.tools.citations import CITATION_RE

FINANCIAL_NUMBER_RE = re.compile(
    r"(?<![\w])\(?-?[$€£]?\d[\d,]*(?:\.\d+)?\)?(?:\s*%)?(?![\w])"
)


def normalize_text(text: str) -> str:
    without_citations = CITATION_RE.sub("", text)
    lowered = without_citations.lower().strip()
    lowered = re.sub(r"\b(the|a|an)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9.%()-]+", " ", lowered)
    return " ".join(lowered.split())


def parse_financial_numbers(text: str) -> list[tuple[Decimal, bool]]:
    cleaned = CITATION_RE.sub("", text).strip()
    parsed: list[tuple[Decimal, bool]] = []
    for match in FINANCIAL_NUMBER_RE.finditer(cleaned):
        raw = match.group(0).strip()
        is_percent = raw.endswith("%")
        if is_percent:
            raw = raw[:-1].strip()
        negative = raw.startswith("(") and raw.endswith(")")
        raw = raw.strip("()").replace(",", "")
        raw = re.sub(r"[$€£]", "", raw)
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        parsed.append((-value if negative else value, is_percent))
    return parsed


def parse_financial_number(text: str) -> tuple[Decimal, bool] | None:
    parsed = parse_financial_numbers(text)
    if not parsed:
        return None
    percentages = [number for number in parsed if number[1]]
    return percentages[-1] if percentages else parsed[-1]


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
