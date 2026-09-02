"""Safe Decimal arithmetic for financial expressions; never calls eval()."""

from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains syntax outside the arithmetic whitelist."""


_PERCENT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%")
_CURRENCY = re.compile(r"[$€£]")


def normalize_expression(expression: str) -> str:
    cleaned = _CURRENCY.sub("", expression).replace(",", "")
    cleaned = _PERCENT.sub(r"(\1 / 100)", cleaned)
    return cleaned.strip()


def _calculate_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _calculate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or abs(right) > 12:
                raise UnsafeExpressionError("Exponent must be an integer between -12 and 12")
            return left**int(right)
    raise UnsafeExpressionError(f"Unsupported syntax: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    normalized = normalize_expression(expression)
    if len(normalized) > 200:
        raise UnsafeExpressionError("Expression is too long")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("Invalid arithmetic syntax") from exc
    try:
        with localcontext() as context:
            context.prec = 28
            # Bound magnitude explicitly so nested exponents cannot escape as a raw
            # decimal.Overflow (an ArithmeticError the graph does not treat as abstention).
            context.Emax = 999
            context.Emin = -999
            result = _calculate_node(tree)
    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as exc:
        raise UnsafeExpressionError("Invalid arithmetic operation") from exc
    except ArithmeticError as exc:  # decimal.Overflow, decimal.Underflow, ...
        raise UnsafeExpressionError("Arithmetic result is out of range") from exc
    if not result.is_finite():
        raise UnsafeExpressionError("Non-finite result")
    rendered = format(result.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"

