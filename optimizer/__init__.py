"""Optimizer module: quantization and accelerator kernels.

This package provides low-level optimizations for inference:
- Quantization: q8_0, q4_0, nf4 formats for KV-cache compression
- Triton kernels: Fused decoupled attention for CUDA

Caramba kernel policy:
- Fast path or exception (no silent fallbacks)
- Capability detection + validation at startup
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimizer.quantizer import Quantizer as Quantizer

__all__ = ["Quantizer"]

# Initialize kernel registry at package import so any missing/invalid kernel backends
# fail loudly before training/inference begins.
from optimizer.kernel_registry import KERNELS as _KERNELS  # noqa: F401


def __getattr__(name: str) -> Any:
    # Lazy import to avoid circular imports (e.g. optimizer <-> cache).
    if name == "Quantizer":
        from optimizer.quantizer import Quantizer

        return Quantizer
    raise AttributeError(name)
