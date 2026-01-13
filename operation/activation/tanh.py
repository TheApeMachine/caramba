"""Hyperbolic tangent activation function

Maps input to (-1, 1) range, commonly used in recurrent neural networks.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.activation.base import ActivationOperation


class TanhOperation(ActivationOperation):
    """Hyperbolic tangent activation

    Maps input values to the range (-1, 1), providing smooth, zero-centered activation
    commonly used in recurrent neural networks and other architectures.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, x: Tensor) -> Tensor:
        """Apply tanh activation

        Computes (exp(x) - exp(-x)) / (exp(x) + exp(-x)), mapping any real value
        to the (-1, 1) range with zero-centered, smooth activation.
        """
        return torch.tanh(x)