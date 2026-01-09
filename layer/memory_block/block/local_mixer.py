"""Local mixer

Implements a causal, depthwise convolution + gated MLP mixer.
This replaces attention's short-range pattern modeling with a fixed-window operator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.layer.memory_block.state import MemoryBlockState


class LocalMixer(nn.Module):
    """Local mixer module

    Local operations are a cheap way to capture short-range patterns; MOSAIC uses
    them to handle “nearby” structure so explicit memory can focus on longer
    horizons.
    """
    def __init__(
        self,
        *,
        conv: nn.Conv1d,
        gate_proj: nn.Linear,
        mlp_up: nn.Linear,
        mlp_down: nn.Linear,
        dropout: nn.Dropout,
        conv_kernel: int,
    ) -> None:
        super().__init__()
        if int(conv_kernel) < 1:
            raise ValueError("conv_kernel must be >= 1")
            
        self.conv = conv
        self.gate_proj = gate_proj
        self.mlp_up = mlp_up
        self.mlp_down = mlp_down
        self.dropout = dropout
        self.conv_kernel = int(conv_kernel)

    def forward(self, u: Tensor, *, state: MemoryBlockState | None) -> tuple[Tensor, Tensor | None]:
        """Compute local features and updated buffer

        The conv buffer carries the last k−1 normalized tokens so single-token
        decoding can use the same causal convolution semantics as training.
        """
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")

        B, T, D = u.shape
        k = int(self.conv_kernel)
        new_buf: Tensor | None = None

        if state is not None and int(T) == 1 and k > 1:
            window = torch.cat([state.conv_buf.to(dtype=u.dtype, device=u.device), u], dim=1)
            x = F.conv1d(
                window.transpose(1, 2),
                self.conv.weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=D,
            ).transpose(1, 2)
            new_buf = window[:, 1:, :].detach()
        else:
            x = self.conv(u.transpose(1, 2))[:, :, :T].transpose(1, 2)
            if state is not None and k > 1:
                keep = min(k - 1, int(T))
                new_buf = u[:, -keep:, :].detach()

        gate = torch.sigmoid(self.gate_proj(u))
        mlp = self.mlp_down(F.silu(self.mlp_up(u)))

        return self.dropout(gate * (x + mlp)), new_buf

