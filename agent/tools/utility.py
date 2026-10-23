"""Small utility tools: safe math evaluation and clock access."""
from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

# Rewrites postfix factorials (5!, (2+3)!) into factorial(...) calls.
_FACTORIAL_RE = re.compile(r"(\d+(?:\.\d+)?|\([^()]*\))!")

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_eval(expression: str) -> float:
    """Evaluate a math expression without executing arbitrary code."""

    def _eval(node: ast.AST):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"unsupported constant: {node.value!r}")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"unknown function: {node.func.id}")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return fn(*[_eval(arg) for arg in node.args])
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"unknown name: {node.id}")
        if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            return [_eval(element) for element in node.elts]
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    prepared = _FACTORIAL_RE.sub(r"factorial(\1)", expression)
    tree = ast.parse(prepared, mode="eval")
    return _eval(tree)


def utility_tools() -> list[BaseTool]:
    """Build the model-independent utility tools."""

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression and return the numeric result.

        Supports + - * / // % ** parentheses, postfix factorial (5!), and
        functions like sqrt, log, sin, cos, tan, factorial, gcd, plus the
        constants pi, e and tau.

        Args:
            expression: The math expression to evaluate.
        """
        try:
            result = safe_eval(expression)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    @tool
    def current_datetime(timezone_name: str = "UTC") -> str:
        """Get the current date and time, optionally in a named IANA timezone.

        Args:
            timezone_name: IANA timezone name, for example "Asia/Kolkata". Defaults to UTC.
        """
        now = datetime.now(timezone.utc)
        if timezone_name and timezone_name.lower() not in {"utc", "gmt"}:
            try:
                now = now.astimezone(ZoneInfo(timezone_name))
            except (ZoneInfoNotFoundError, ValueError) as exc:
                return f"ERROR: unknown timezone '{timezone_name}': {exc}"
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")

    return [calculate, current_datetime]
