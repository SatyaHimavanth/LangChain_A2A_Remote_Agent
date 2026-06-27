from __future__ import annotations

from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator


def _coerce_number(value: Any) -> float:
    if isinstance(value, dict):
        if "$exp" in value and "$number" in value:
            return float(pow(_coerce_number(value["$exp"]), _coerce_number(value["$number"])))
        if "base" in value and "exponent" in value:
            return float(pow(_coerce_number(value["base"]), _coerce_number(value["exponent"])))

    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid calculator numbers.")

    return float(value)


def _format_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


class BinaryOperationInput(BaseModel):
    a: float = Field(description="The first number.")
    b: float = Field(description="The second number.")

    @field_validator("a", "b", mode="before")
    @classmethod
    def coerce_numbers(cls, value: Any) -> float:
        return _coerce_number(value)


class PowerInput(BaseModel):
    base: float = Field(description="The base number, the left side of ^.")
    exponent: float = Field(description="The exponent, the right side of ^.")

    @field_validator("base", "exponent", mode="before")
    @classmethod
    def coerce_numbers(cls, value: Any) -> float:
        return _coerce_number(value)


class RootInput(BaseModel):
    value: float = Field(description="The number to take the root of.")
    index: float = Field(default=2, description="The root index. Use 2 for square root.")

    @field_validator("value", "index", mode="before")
    @classmethod
    def coerce_numbers(cls, value: Any) -> float:
        return _coerce_number(value)


@tool(args_schema=BinaryOperationInput)
def addition(a: float, b: float) -> int | float:
    """Add two numbers."""
    return _format_number(a + b)


@tool(args_schema=BinaryOperationInput)
def subtraction(a: float, b: float) -> int | float:
    """Subtract the second number from the first."""
    return _format_number(a - b)


@tool(args_schema=BinaryOperationInput)
def multiplication(a: float, b: float) -> int | float:
    """Multiply two numbers."""
    return _format_number(a * b)


@tool(args_schema=BinaryOperationInput)
def division(a: float, b: float) -> int | float:
    """Divide the first number by the second."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero. Please provide a non-zero divisor.")
    return _format_number(a / b)


@tool(args_schema=PowerInput)
def power(base: float, exponent: float) -> int | float:
    """Raise the base to the exponent. Use this for caret expressions like 2^6."""
    if base == 0 and exponent < 0:
        raise ValueError("0 cannot be raised to a negative power.")
    return _format_number(pow(base, exponent))


@tool(args_schema=RootInput)
def root(value: float, index: float = 2) -> int | float:
    """Calculate the n-th root of a value. Default index to 2 for square root."""
    if index == 0:
        raise ZeroDivisionError("The root index cannot be zero.")
    if value == 0 and index < 0:
        raise ValueError("Cannot calculate a negative root of zero.")
    if value < 0 and not float(index).is_integer():
        raise ValueError("Cannot calculate a fractional root of a negative number.")
    return _format_number(pow(value, 1 / index))
