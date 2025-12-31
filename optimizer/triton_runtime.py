"""Compatibility shim for Triton runtime detection.

This project originally exposed Triton availability via `optimizer.triton_runtime`.
The Metal/MPS fused DBA work requires backend-agnostic detection, which now lives in
`optimizer.runtime`.

Keep this module to avoid breaking imports.
"""

from __future__ import annotations

from caramba.optimizer.runtime import (
    TRITON_AVAILABLE,
    triton_decoupled_q4q8q4_available,
    triton_ssm_available,
)

__all__ = [
    "TRITON_AVAILABLE",
    "triton_decoupled_q4q8q4_available",
    "triton_ssm_available",
]
