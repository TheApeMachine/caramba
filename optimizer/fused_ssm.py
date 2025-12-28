"""Fused SSM operations using Triton kernels.

This module provides optimized SSM implementations that use Triton
kernels on CUDA for high-throughput sequence processing.
"""
from __future__ import annotations

import torch
from torch import Tensor

from optimizer.triton_runtime import TRITON_AVAILABLE
from optimizer.kernels_ssm import selective_scan_triton

__all__ = ["fused_selective_scan", "fused_ssm_available"]


def fused_ssm_available(device_type: str) -> bool:
    """Check if fused SSM kernels are available."""
    return TRITON_AVAILABLE and device_type == "cuda"


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
        raise RuntimeError("Fused SSM kernels not available or not on CUDA")

    # The triton kernel handles the batch/dim flattening
    return selective_scan_triton(x, dt, A, B, C, D)
