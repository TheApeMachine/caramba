"""Reusable math utilities (named techniques) used across the framework.

This exists to remove duplicated arithmetic from higher-level code and to
give common heuristics a *name*.
"""

from __future__ import annotations

from .splits import train_val_counts
from .precision import autocast_dtype
from .precision import autocast_dtype_str, weight_dtype, weight_dtype_str
from .optim import global_grad_norm_l2
from .bytes import bytes_per_kind
from .batching import token_budget_batch_size
from .sketch import stable_int_hash, stride_sketch_indices, sketch_dot5
from .linalg import randomized_svd

__all__ = [
    "train_val_counts",
    "autocast_dtype",
    "autocast_dtype_str",
    "weight_dtype",
    "weight_dtype_str",
    "global_grad_norm_l2",
    "bytes_per_kind",
    "token_budget_batch_size",
    "stable_int_hash",
    "stride_sketch_indices",
    "sketch_dot5",
    "randomized_svd",
]

