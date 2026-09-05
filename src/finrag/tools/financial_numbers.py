"""Explicit financial quantities in base units; no implicit scale or FX conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

SCALES = {
    "thousand": Decimal(1000),
    "million": Decimal(1000000),
    "billion": Decimal(1000000000),
    "trillion": Decimal(1000000000000),
    "mn": Decimal(1000000),
    "bn": Decimal(1000000000),
}
CURRENCIES = {
    "$": "USD",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
}
QUANTITY_RE = re.compile(
    r"(?<![\w.])(?P<open>\()?\s*(?P<sign>[-+−])?\s*"
    r"(?P<prefix>[$€£]|USD\b|EUR\b|GBP\b)?\s*"
    r"(?P<number>(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<close>\))?(?:\s*(?P<scale>thousand|million|billion|trillion|mn|bn)\b)?"
    r"(?:\s*(?P<unit>%|percent\b|USD\b|EUR\b|GBP\b|dollars?\b|euros?\b|pounds?\b))?"
    r"(?P<end_close>\))?(?![\w])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FinancialQuantity:
    amount: Decimal
    scale: Decimal
    is_percent: bool
    currency: str | None
    raw: str

    @property
    def value(self) -> Decimal:
        return self.amount * self.scale


def parse_quantities(text: str) -> list[FinancialQuantity]:
    cleaned = re.sub(r"\[CITATION:\s*[^\]\n]+\]", "", text)
    quantities = []
    for match in QUANTITY_RE.finditer(cleaned):
        amount = Decimal(match["number"].replace(",", ""))
        if match["sign"] in {"-", "−"} or (
            match["open"] and (match["close"] or match["end_close"])
        ):
            amount = -amount
        prefix = (match["prefix"] or "").lower()
        unit = (match["unit"] or "").lower()
        currency = CURRENCIES.get(prefix) or CURRENCIES.get(unit)
        if prefix in CURRENCIES and unit in CURRENCIES and CURRENCIES[prefix] != CURRENCIES[unit]:
            currency = "CONFLICT"
        quantities.append(
            FinancialQuantity(
                amount=amount,
                scale=SCALES.get((match["scale"] or "").lower(), Decimal(1)),
                is_percent=unit in {"%", "percent"},
                currency=currency,
                raw=match.group(0).strip(),
            )
        )
    return quantities


def answer_quantity(text: str) -> FinancialQuantity | None:
    quantities = parse_quantities(text)
    percentages = [quantity for quantity in quantities if quantity.is_percent]
    return (percentages or quantities)[-1] if quantities else None


def compatible_units(left: FinancialQuantity, right: FinancialQuantity) -> bool:
    # FinQA often omits currency in gold labels; absence is unspecified, never FX.
    return (
        left.is_percent == right.is_percent
        and "CONFLICT" not in {left.currency, right.currency}
        and (left.currency is None or right.currency is None or left.currency == right.currency)
    )
