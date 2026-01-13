"""Attention masking operations

Operations for creating and applying masks in attention mechanisms,
ensuring causal constraints and preventing information leakage.
"""
from __future__ import annotations

from caramba.operation.masking.apply_mask import ApplyMaskOperation
from caramba.operation.masking.base import MaskingOperation
from caramba.operation.masking.causal_mask_like import CausalMaskLikeOperation
from caramba.operation.masking.causal_mask import CausalMaskOperation
from caramba.operation.masking.combine_masks import CombineMasksOperation

__all__ = [
    "MaskingOperation",
    "ApplyMaskOperation",
    "CausalMaskOperation",
    "CausalMaskLikeOperation",
    "CombineMasksOperation",
]
