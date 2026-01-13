"""Shape operation base classes

Foundation classes for tensor shape manipulation operations in neural networks.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class ShapeOperation(Operation):
    """Base class for all shape transformation operations

    Provides common interface for operations that change tensor dimensions,
    used extensively in attention mechanisms and neural network architectures.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Transform tensor shape

        Subclasses implement specific shape transformations like reshaping,
        transposing, or splitting tensors for different neural network operations.
        """
        raise NotImplementedError("Subclasses must implement forward pass.")