"""Scaled dot-product attention operation

Core attention mechanism computing weighted combinations of values based on query-key similarities.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from caramba.operation.attention.base import AttentionOperation


class ScaledDotProductAttentionOperation(AttentionOperation):
    """Scaled dot-product attention

    Computes attention by taking dot products between queries and keys,
    scaling by sqrt(d_k), applying softmax, and weighted sum of values.
    """
    def __init__(self, *, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout_p = dropout_p

    def forward(
        self,
        *,
        q: Tensor,  # [batch, num_heads, seq_len, head_dim]
        k: Tensor,  # [batch, num_heads, seq_len, head_dim]
        v: Tensor,  # [batch, num_heads, seq_len, head_dim]
        mask: Tensor | None = None,  # [batch, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
    ) -> Tensor:
        """Compute scaled dot-product attention

        Takes queries, keys, and values and computes attention weights,
        optionally applying a mask and dropout for regularization.
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        head_dim = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape if needed
            if mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout if specified
        if self.dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # Weighted sum of values: attention_weights * V
        output = torch.matmul(attn_weights, v)

        return output