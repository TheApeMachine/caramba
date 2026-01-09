"""GPT-2 style initialization."""

from __future__ import annotations

import torch
from torch import nn
from caramba.initializers.base import Initializer

class GPT2Initializer(Initializer):
    """Initializes weights with standard GPT-2 style scaling.

    PyTorch default initialization is too large for deep transformers,
    often causing instability/spikes during warmup. This initializer:
    - Uses Normal(0, 0.02) for weights
    - Scales down residual projections by 1/sqrt(2 * n_layers)
    - Zeros biases
    """

    def __init__(self, n_layers: int = 12) -> None:
        self.n_layers = int(n_layers)

    def initialize(self, module: nn.Module) -> None:
        """Initialize the weights of the given module (in-place)."""

        # Apply initialization
        for name, p in module.named_parameters():
            if "weight" in name and p.dim() >= 2:
                # Standard init for all weights
                nn.init.normal_(p, mean=0.0, std=0.02)

                # Scale down residual projections
                # Attention output: out_proj
                # MLP output: down_proj (SwiGLU) or c_proj (GPT2)
                if "out_proj" in name or "down_proj" in name or "c_proj" in name:
                    scale = 1.0 / (2.0 * self.n_layers)**0.5
                    p.data.mul_(scale)

            elif "bias" in name:
                nn.init.zeros_(p)
            elif "logit_scale" in name:
                # Learned attention temperature (if any)
                nn.init.zeros_(p)

        # Specific override for norms (if they are parameters)
        for m in module.modules():
            # Support both standard PyTorch norms and any custom ones that expose weight/bias
            # Check class name string to avoid importing specific layers
            if "Norm" in type(m).__name__:
                if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor):
                    nn.init.zeros_(m.bias)
