"""Merge attention heads operation

Combines multiple attention heads back into a single representation.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class MergeHeadsOperation(Operation):
    """Combine attention heads into single tensor

    Reverses the head-splitting operation by merging multiple attention heads
    back into a unified hidden representation for further processing.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, x: Tensor) -> Tensor:
        """Merge heads into single representation

        Transposes and reshapes tensor from [batch, num_heads, seq_len, head_dim]
        back to [batch, seq_len, hidden_dim] by combining all attention heads.
        """
        # x: [batch, num_heads, seq_len, head_dim]
        # transpose(1, 2): [batch, seq_len, num_heads, head_dim]
        # view: [batch, seq_len, num_heads * head_dim]
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)