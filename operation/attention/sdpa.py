"""PyTorch scaled dot-product attention (SDPA) operation.

This wraps `torch.nn.functional.scaled_dot_product_attention` so op-graphs can
use the optimized backend while keeping a manifest-friendly interface.

Mask semantics match `caramba.operation.masking`:
- Boolean `mask` uses True for "masked out / do not attend".

Internally, PyTorch SDPA expects the inverse boolean convention (True = keep),
so we invert boolean masks before calling into PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from caramba.operation.attention.base import AttentionOperation


class SDPAOperation(AttentionOperation):
    """Scaled dot-product attention via PyTorch SDPA."""

    def __init__(
        self,
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.is_causal = bool(is_causal)
        self.scale = float(scale) if scale is not None else None

    def forward(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        attn_mask = mask
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # Op-level convention: True means "mask out". SDPA expects True means "keep".
            attn_mask = ~attn_mask

        dropout_p = float(self.dropout_p) if self.training else 0.0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=bool(self.is_causal),
            scale=self.scale,
        )

