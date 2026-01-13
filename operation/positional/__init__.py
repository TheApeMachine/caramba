"""Positional encoding operations

Operations for injecting positional information into sequences,
essential for models to understand token ordering and relative positions.
"""
from __future__ import annotations

from caramba.operation.positional.apply_rope import ApplyRoPEOperation
from caramba.operation.positional.base import PositionalOperation

__all__ = [
    "PositionalOperation",
    "ApplyRoPEOperation",
]