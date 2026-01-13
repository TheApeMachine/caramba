"""Causal masking for autoregressive attention

Creates masks that prevent attending to future tokens.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.masking.base import MaskingOperation


class CausalMaskOperation(MaskingOperation):
    """Generate causal attention mask

    Creates a triangular mask that prevents attention to future positions,
    ensuring autoregressive models can only attend to past and current tokens.
    """
    def __init__(self, *, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, *, batch_size: int = 1) -> Tensor:
        """Create causal mask

        Generates a boolean mask where position (i,j) is True if j <= i,
        meaning position i can attend to positions j where j is not in the future.
        """
        # Create causal mask: True for positions that should be masked (not attended to)
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool), diagonal=1)
        # Expand for batch dimension
        if batch_size > 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask