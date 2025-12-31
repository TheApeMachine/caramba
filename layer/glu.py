"""Gated Linear Unit (GLU) implementation.

GLU is a common MLP variant that uses a gate to control information flow.
It computes output = down(act(gate(x)) * up(x)). Common activations
include SiLU (SwiGLU), GELU (GeGLU), and ReLU (ReGLU). This implementation
uses a fused projection for the gate and up matrices to maximize hardware
utilization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import GLULayerConfig


class GLULayer(nn.Module):
    """Gated Linear Unit layer with fused projections."""

    def __init__(self, config: GLULayerConfig) -> None:
        """Initialize the projections and activation.

        Args:
            config: Specifies dimensions, activation type, and bias.
        """
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.d_ff = int(config.d_ff)
        self.bias = bool(config.bias)

        # Fuse gate and up projections into one matrix
        self.w_gate_up = nn.Linear(self.d_model, 2 * self.d_ff, bias=self.bias)
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=self.bias)

        self.activation_name = config.activation.lower()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply GLU: act(gate(x)) * up(x), then down-project."""
        # Fused projection: (B, T, d_model) @ (d_model, 2 * d_ff) -> (B, T, 2 * d_ff)
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # Apply activation and combine
        return self.w_down(self._apply_activation(gate) * up)

    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply the configured activation function."""
        if self.activation_name == "silu":
            return F.silu(x)
        elif self.activation_name == "gelu":
            return F.gelu(x)
        elif self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
