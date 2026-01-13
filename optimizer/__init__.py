"""Optimizer compatibility layer.

The caramba codebase is transitioning from the historical `caramba.optimizer`
namespace to `caramba.kernel` (which holds the actual implementations for
quantization and accelerator kernels).

This package keeps legacy import paths working while the refactor is in
progress, without duplicating implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from caramba.kernel.quantizer import Quantizer as Quantizer

__all__ = [
    "Quantizer",
    "KERNELS",
    "initialize_kernels",
]


def __getattr__(name: str) -> Any:
    """Lazily forward optimizer symbols to `caramba.kernel`."""
    if name == "Quantizer":
        from caramba.kernel.quantizer import Quantizer

        return Quantizer
    if name == "KERNELS":
        from caramba.kernel.kernel_registry import KERNELS

        return KERNELS
    if name == "initialize_kernels":
        from caramba.kernel.kernel_registry import initialize_kernels

        return initialize_kernels
    raise AttributeError(name)

