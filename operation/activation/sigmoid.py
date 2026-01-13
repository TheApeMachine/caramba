"""Sigmoid activation function

Logistic activation function that squashes input to (0, 1) range.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.activation.base import ActivationOperation
from typing_extensions import override


class SigmoidOperation(ActivationOperation):
    """Sigmoid activation

    Maps input values to the range (0, 1), commonly used for binary classification
    and as gating mechanisms in neural networks.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, *, x: Tensor) -> Tensor:
        """Apply sigmoid activation

        Computes 1 / (1 + exp(-x)), mapping any real value to the (0, 1) range
        which is useful for probability outputs and gating mechanisms.
        """
        return torch.sigmoid(x)