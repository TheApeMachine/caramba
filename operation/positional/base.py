"""Positional encoding operation base classes

Foundation classes for positional encoding operations in neural networks.
"""
from __future__ import annotations

from caramba.operation.base import Operation


class PositionalOperation(Operation):
    """Base class for all positional encoding operations

    Provides common interface for operations that inject positional information
    into sequences, essential for models to understand token ordering.
    """
    def __init__(self) -> None:
        super().__init__()