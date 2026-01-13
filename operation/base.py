"""Base operation class for neural networks

Foundation for all composable operations in neural network architectures.
"""
from __future__ import annotations

from typing_extensions import override
from torch import Tensor, nn


class Operation(nn.Module):
    """Base class for neural network operations

    Provides common interface for all operations that transform tensors,
    enabling composition and reuse across different model architectures.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, *, x: Tensor) -> Tensor:
        """Transform input tensor

        Defines the core computation performed by this operation.
        Subclasses implement specific transformations like attention, convolution, or normalization.
        """
        raise NotImplementedError("Subclasses must implement forward pass.")