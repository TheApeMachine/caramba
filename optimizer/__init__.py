"""Optimizer module: quantization and optional Triton kernels.

This package provides low-level optimizations for inference:
- Quantization: q8_0, q4_0, nf4 formats for KV-cache compression
- Triton kernels: Fused decoupled attention for CUDA
- Runtime checks: Safe fallbacks when Triton isn't available
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimizer.quantizer import Quantizer as Quantizer

__all__ = ["Quantizer"]


def __getattr__(name: str) -> Any:
    # Lazy import to avoid circular imports (e.g. optimizer <-> cache).
    if name == "Quantizer":
        from optimizer.quantizer import Quantizer

        return Quantizer
    raise AttributeError(name)
