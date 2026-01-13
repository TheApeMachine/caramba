"""Gaussian Error Linear Unit activation function

Smooth approximation of ReLU used in many modern transformer architectures.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.activation.base import ActivationOperation


class GELUOperation(ActivationOperation):
    """Gaussian Error Linear Unit activation

    Smooth, differentiable approximation of ReLU that performs well in practice,
    commonly used in transformer models like BERT and GPT.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, x: Tensor) -> Tensor:
        """Apply GELU activation

        Computes 0.5 * x * (1 + erf(x/sqrt(2))), providing smooth non-linearity
        that outperforms ReLU in many transformer architectures.
        """
        return torch.nn.functional.gelu(x)