"""Rectified Linear Unit activation function

The most fundamental activation function in deep learning.
"""
from __future__ import annotations

import torch
from typing_extensions import override
from torch import Tensor

from caramba.operation.activation.base import ActivationOperation


class ReLUOperation(ActivationOperation):
    """Rectified Linear Unit activation

    Sets all negative values to zero while keeping positive values unchanged,
    introducing non-linearity that allows networks to learn complex patterns.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, *, x: Tensor) -> Tensor:
        """Apply ReLU activation

        Returns max(0, x) element-wise, effectively "rectifying" negative values to zero.
        """
        return torch.relu(x)