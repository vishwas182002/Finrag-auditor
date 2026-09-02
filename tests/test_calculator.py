from __future__ import annotations

import pytest

from finrag.tools.calculator import UnsafeExpressionError, normalize_expression, safe_calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(120 - 100) / 100 * 100", "20"),
        ("$1,200 + 300", "1500"),
        ("12.5% * 200", "25"),
        ("2 ** 3", "8"),
    ],
)
def test_safe_calculator(expression: str, expected: str) -> None:
    assert safe_calculate(expression) == expected


def test_percentage_normalization() -> None:
    assert normalize_expression("12.5% + 1") == "(12.5 / 100) + 1"


@pytest.mark.parametrize("expression", ["__import__('os')", "open('/tmp/x')", "2 ** 100"])
def test_unsafe_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_calculate(expression)

