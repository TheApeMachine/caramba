"""Compatibility shim for Triton runtime detection.

This project originally exposed Triton availability via `optimizer.triton_runtime`.
The Metal/MPS fused DBA work requires backend-agnostic detection, which now lives in
`optimizer.runtime`.

Keep this module to avoid breaking imports.
"""

from __future__ import annotations

from optimizer.runtime import (
    triton_supported,
)

__all__ = [
    "triton_supported",
]
