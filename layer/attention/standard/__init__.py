"""Standard attention variants

This package contains the “vanilla” attention implementation (including GQA),
which is the baseline most other attention ideas are compared against.
"""
from __future__ import annotations

from caramba.layer.attention.standard.layer import StandardAttentionLayer

__all__ = ["StandardAttentionLayer"]
