"""Optimizer runtime helpers (compatibility shim).

This module re-exports backend availability checks from `caramba.kernel.runtime`
so legacy imports (`caramba.optimizer.runtime`) continue to work.
"""

from __future__ import annotations

from caramba.kernel.runtime import (
    has_module,
    metal_build_tools_available,
    metal_supported,
    triton_supported,
)

__all__ = [
    "has_module",
    "metal_build_tools_available",
    "metal_supported",
    "triton_supported",
]

