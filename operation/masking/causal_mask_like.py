"""Dynamic causal mask from query/key tensors.

Unlike `CausalMaskOperation`, this variant derives the mask shape from the
provided Q/K tensors, making it usable in op graphs where sequence lengths can
vary (e.g. null-token injection or cache-assisted decode).

Mask semantics follow the rest of `caramba.operation.masking`:
- Boolean mask where True means "masked out / do not attend".
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.masking.base import MaskingOperation


class CausalMaskLikeOperation(MaskingOperation):
    """Create a causal mask shaped like (T_q, T_k)."""

    def __init__(self, *, diagonal: int = 1) -> None:
        super().__init__()
        self.diagonal = int(diagonal)

    def forward(self, *, q: Tensor, k: Tensor) -> Tensor:
        if q.dim() < 2 or k.dim() < 2:
            raise ValueError("q and k must be at least rank-2 (.., T, D)")
        t_q = int(q.size(-2))
        t_k = int(k.size(-2))
        if t_q <= 0 or t_k <= 0:
            raise ValueError(f"Expected T_q,T_k > 0, got {t_q},{t_k}")

        # True means "masked out".
        base = torch.ones((t_q, t_k), device=q.device, dtype=torch.bool)
        mask = torch.triu(base, diagonal=int(self.diagonal))
        # Broadcast-friendly shape: [1, 1, T_q, T_k]
        return mask.view(1, 1, t_q, t_k)

