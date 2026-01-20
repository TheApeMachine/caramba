"""Gated Linear Unit (GLU) layer

GLU-style MLPs learn a “gate” that modulates an “up” projection, which often
improves expressiveness per parameter compared to a plain MLP while keeping the
math friendly to fused matrix multiplies.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from config.layer import GLULayerConfig


class GLULayer(nn.Module):
    """Gated Linear Unit layer

    The fused gate+up projection is a practical performance trick: it turns two
    linear layers into one larger GEMM, which tends to be faster on modern
    accelerators.
    """

    def __init__(self, config: GLULayerConfig) -> None:
        """Initialize GLU projections

        GLU splits one projection into two halves: one half becomes the gate,
        the other becomes the values being gated.
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
        """Apply GLU

        Gating is a simple mechanism that lets the network choose which features
        to amplify or suppress on a per-token basis.
        """
        # Fused projection: (B, T, d_model) @ (d_model, 2 * d_ff) -> (B, T, 2 * d_ff)
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # Apply activation and combine
        return self.w_down(self._apply_activation(gate) * up)

    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply the configured activation

        Different GLU variants are mostly “which nonlinearity is in the gate”;
        keeping it configurable makes comparisons easy.
        """
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
