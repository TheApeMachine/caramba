"""Swish-Gated Linear Unit activation function

Gated activation function used in modern transformer architectures like Llama.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.activation.base import ActivationOperation
from typing_extensions import override


class SwiGLUOperation(ActivationOperation):
    """Swish-Gated Linear Unit activation

    Gated activation that splits input into two halves, applies Swish to one half
    and uses it to gate the other half, commonly used in feed-forward networks.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, *, x: Tensor) -> Tensor:
        """Apply SwiGLU activation

        Splits input tensor in half along last dimension, applies gating mechanism
        where one half controls (gates) the output of the other half using Swish activation.
        """
        # Split input into two halves along last dimension
        x1, x2 = x.chunk(2, dim=-1)
        # Apply gating: x1 * swish(x2)
        return x1 * torch.nn.functional.silu(x2)