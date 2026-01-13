"""Split tensor into attention heads

Prepares tensors for multi-head attention by dividing the hidden dimension across multiple heads.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class ViewAsHeadsOperation(Operation):
    """Split hidden dimension into attention heads

    Divides the hidden dimension of a tensor into multiple attention heads,
    enabling parallel attention computation across different representation subspaces.
    """
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        self.num_heads = num_heads

    def forward(self, *, x: Tensor) -> Tensor:
        """Reshape tensor for multi-head processing

        Splits the last dimension of input tensor into multiple heads,
        transforming [batch, seq_len, hidden_dim] into [batch, seq_len, num_heads, head_dim].
        """
        batch_size, seq_len, hidden_dim = x.shape
        head_dim = hidden_dim // self.num_heads

        if hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})")

        # Reshape to [batch, seq_len, num_heads, head_dim]
        return x.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)