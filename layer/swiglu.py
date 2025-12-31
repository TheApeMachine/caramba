"""SwiGLU: the MLP variant used in Llama and modern transformers.

SwiGLU combines a gated linear unit with the SiLU (Swish) activation.
The gate and up projections are computed in parallel, then combined:
output = down(silu(gate(x)) * up(x)). This implementation uses a fused
projection for the gate and up matrices to maximize hardware utilization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import SwiGLULayerConfig


class SwiGLULayer(nn.Module):
    """SwiGLU MLP layer with fused gate/up projections.

    The hidden dimension (d_ff) is typically 8/3 × d_model. This implementation
    fuses the gate and up projections into a single linear layer to improve
    throughput during both training and inference.
    """

    def __init__(self, config: SwiGLULayerConfig) -> None:
        """Initialize the projections.

        Args:
            config: Specifies d_model, d_ff (hidden dim), and bias settings.
        """
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.d_ff = int(config.d_ff)
        self.bias = bool(config.bias)

        # Fuse gate and up projections into one matrix
        self.w_gate_up = nn.Linear(self.d_model, 2 * self.d_ff, bias=self.bias)
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=self.bias)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply SwiGLU: silu(gate(x)) * up(x), then down-project.

        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            Output tensor (B, T, d_model)
        """
        # Fused projection: (B, T, d_model) @ (d_model, 2 * d_ff) -> (B, T, 2 * d_ff)
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # Element-wise operations
        return self.w_down(F.silu(gate) * up)
