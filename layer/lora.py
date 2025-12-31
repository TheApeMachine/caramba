"""LoRA (Low-Rank Adaptation) linear layer implementation.

LoRA injects low-rank matrices into frozen layers to enable efficient fine-tuning.
Our implementation is a standalone layer that can be used in manifests
anywhere a linear projection is needed.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import LoRALinearLayerConfig


class LoRALinearLayer(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA).

    Computes: output = x @ W + (x @ A @ B) * (alpha / r)
    where W is the frozen weight and A, B are the low-rank adaptation matrices.
    """

    def __init__(self, config: LoRALinearLayerConfig) -> None:
        """Initialize the base linear layer and LoRA adapters.

        Args:
            config: Specifies input/output dims, rank (r), alpha, and dropout.
        """
        super().__init__()
        self.config = config
        self.d_in = config.d_in
        self.d_out = config.d_out
        self.r = config.r
        self.alpha = config.alpha
        self.scaling = config.alpha / config.r

        # Base linear layer (typically frozen during training)
        self.linear = nn.Linear(self.d_in, self.d_out, bias=config.bias)

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros((self.d_in, self.r)))
        self.lora_B = nn.Parameter(torch.zeros((self.r, self.d_out)))
        self.lora_dropout = nn.Dropout(p=config.dropout)

        # Initialize LoRA parameters
        # A is typically Kaiming-uniform, B is zeros so the adapter starts as identity (0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply base projection + low-rank adaptation."""
        # x: (..., d_in)
        result = self.linear(x)

        # Apply LoRA path
        # (..., d_in) @ (d_in, r) @ (r, d_out) -> (..., d_out)
        adapter = (self.lora_dropout(x) @ self.lora_A) @ self.lora_B

        return result + adapter * self.scaling
