"""Fused SSM operations using Triton kernels.

This module provides optimized SSM implementations that use Triton
kernels on CUDA for high-throughput sequence processing.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.kernel.runtime import triton_supported
from caramba.kernel.kernels_ssm import selective_scan_triton

__all__ = ["fused_selective_scan", "fused_ssm_available"]


def fused_ssm_available(device_type: str) -> bool:
    """Check if fused SSM kernels are available."""
    return bool(triton_supported() and device_type == "cuda" and selective_scan_triton is not None)


def fused_selective_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
) -> Tensor:
    """Run the fused selective scan kernel.

    Args:
        x: Input features (B, T, D_inner)
        dt: Time steps (B, T, D_inner)
        A: State transition matrix (D_inner, D_state)
        B: Input-to-state matrix (B, T, D_state)
        C: State-to-output matrix (B, T, D_state)
        D: Skip connection (D_inner)

    Returns:
        Output features (B, T, D_inner)
    """
    if not fused_ssm_available(x.device.type) or selective_scan_triton is None:
        raise RuntimeError(
            "Fused CUDA SSM kernels are unavailable.\n"
            "This build requires Triton selective-scan kernels for CUDA.\n"
            "Ensure Triton is installed and the SSM kernels import/JIT successfully.\n"
            "This is a hard failure under the kernel policy."
        )

    # The triton kernel handles the batch/dim flattening
    return selective_scan_triton(x, dt, A, B, C, D)
