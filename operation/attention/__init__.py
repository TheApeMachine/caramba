"""Attention mechanism operations

Core attention computations that form the foundation of transformer architectures,
including scaled dot-product attention and attention variants.
"""
from __future__ import annotations

from caramba.operation.attention.base import AttentionOperation
from caramba.operation.attention.sdpa import SDPAOperation
from caramba.operation.attention.scaled_dot_product import ScaledDotProductAttentionOperation

__all__ = [
    "AttentionOperation",
    "SDPAOperation",
    "ScaledDotProductAttentionOperation",
]
