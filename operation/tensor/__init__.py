"""Tensor source/buffer operations.

These operations produce tensors rather than transforming an input tensor.
They are useful in op-graphs for learnable constants (e.g. null-attention tokens)
and other parameterized buffers.
"""
from __future__ import annotations

from caramba.operation.tensor.parameter import ParameterOperation

__all__ = [
    "ParameterOperation",
]

