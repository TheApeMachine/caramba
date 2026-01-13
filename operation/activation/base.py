"""Activation function operation base classes

Foundation classes for activation function operations in neural networks.
"""
from __future__ import annotations

from typing_extensions import override
from torch import Tensor

from caramba.operation.base import Operation


class ActivationOperation(Operation):
    """Base class for all activation operations

    Provides common interface for non-linear activation functions that introduce
    non-linearity into neural networks, enabling complex pattern learning.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, *, x: Tensor) -> Tensor:
        """Forward pass

        Forwards the input tensor and returns the activated tensor.
        """
        raise NotImplementedError("Subclasses must implement forward pass.")