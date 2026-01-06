"""SwiGLU layer

SwiGLU is a GLU-style MLP that uses SiLU in the gate, and it is widely used in
modern LLMs because it tends to give strong quality-per-parameter while staying
friendly to fused matmul kernels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import SwiGLULayerConfig


class SwiGLULayer(nn.Module):
    """SwiGLU MLP layer

    The gate and up projections are fused into one larger GEMM, which is usually
    faster than two separate projections on GPUs and Apple Silicon.
    """

    def __init__(self, config: SwiGLULayerConfig) -> None:
        """Initialize SwiGLU projections

        SwiGLU is structurally “two projections, elementwise gate, then down”;
        fusing the first two projections keeps it simple and fast.
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
        """Apply SwiGLU

        The nonlinearity lives in the gate; multiplying by the “up” projection
        gives the block a multiplicative interaction that plain MLPs lack.
        """
        # Fused projection: (B, T, d_model) @ (d_model, 2 * d_ff) -> (B, T, 2 * d_ff)
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # Element-wise operations
        return self.w_down(F.silu(gate) * up)
