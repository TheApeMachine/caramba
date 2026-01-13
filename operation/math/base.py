"""Mathematical operation base classes

Foundation classes for mathematical operations in neural networks.
"""
from __future__ import annotations

from caramba.operation.base import Operation


class MathOperation(Operation):
    """Base class for all mathematical operations

    Provides common interface for operations involving mathematical computations,
    transformations, and activation functions used throughout neural networks.
    """
    def __init__(self) -> None:
        super().__init__()