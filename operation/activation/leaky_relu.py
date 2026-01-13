"""Leaky Rectified Linear Unit activation function

Variant of ReLU that allows small negative values to pass through.
"""
from __future__ import annotations

import torch
from typing_extensions import override
from torch import Tensor

from caramba.operation.activation.base import ActivationOperation


class LeakyReLUOperation(ActivationOperation):
    """Leaky Rectified Linear Unit activation

    Variant of ReLU that allows a small fraction of the negative values to pass through,
    helping prevent dead neurons and providing better gradient flow.
    """
    def __init__(self, *, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    @override
    def forward(self, *, x: Tensor) -> Tensor:
        """Apply LeakyReLU activation

        Returns x if x > 0, otherwise returns negative_slope * x,
        allowing small negative gradients to flow through the network.
        """
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)