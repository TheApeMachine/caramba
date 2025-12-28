"""Metal (MPS) fused kernels for Apple Silicon.

This submodule provides a serious, real implementation of a fused DBA (decoupled)
attention *decode* path for MPS, backed by a custom Metal Shading Language kernel
and an Objective-C++ PyTorch extension.

The kernel is intended for autoregressive decode (single-token query) where the
primary cost is repeatedly materializing score tensors and performing unfused
softmax/value matmuls. The Metal kernel performs a numerically-stable, two-pass
softmax (max + exp-sum) fused with the V-weighted reduction, without materializing
the full score matrix.
"""

from __future__ import annotations

from .dba_decode import dba_decode_fp16, metal_dba_decode_available
from .rmsnorm import metal_rmsnorm_available, rmsnorm_fp16
from .rope import metal_rope_available, rope_fp16
from .lion import lion_fp16, metal_lion_available

__all__ = [
    "dba_decode_fp16",
    "metal_dba_decode_available",
    "rmsnorm_fp16",
    "metal_rmsnorm_available",
    "rope_fp16",
    "metal_rope_available",
    "lion_fp16",
    "metal_lion_available",
]

