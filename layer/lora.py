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

from config.layer import LoRALinearLayerConfig


class LoRALinearLayer(nn.Module):
    """Low-Rank Adaptation (LoRA) projection

    LoRA is a parameter-efficient trick: instead of updating a full weight
    matrix, you learn a low-rank update that can be trained quickly and stored
    cheaply.
    """

    def __init__(self, config: LoRALinearLayerConfig) -> None:
        """Initialize LoRA adapters

        The rank controls how expressive the update is; scaling by alpha/r keeps
        update magnitudes predictable across different ranks.
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
        """Apply LoRA projection

        You can think of this as “base linear + learned delta”; when the base is
        frozen, LoRA focuses learning capacity where it matters without touching
        the full parameter set.
        """
        # x: (..., d_in)
        result = self.linear(x)

        # Apply LoRA path
        # (..., d_in) @ (d_in, r) @ (r, d_out) -> (..., d_out)
        adapter = (self.lora_dropout(x) @ self.lora_A) @ self.lora_B

        return result + adapter * self.scaling
