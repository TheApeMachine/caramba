"""Public entrypoint for CUDA/Triton SSM kernels.

This module stays intentionally small (size limits): the Triton implementation
is split across `optimizer/ssm_triton.py` and `optimizer/ssm_triton_kernels_{fwd,bwd}.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from caramba.console import logger
from caramba.optimizer.triton_runtime import TRITON_AVAILABLE

__all__ = ["selective_scan_triton"]

selective_scan_triton: Callable | None = None

if not TYPE_CHECKING and TRITON_AVAILABLE:
    try:  # pyright: ignore[reportUnreachable]
        from caramba.optimizer.ssm_triton import selective_scan_triton as _selective_scan_triton
    except Exception as e:
        logger.error(f"Failed to import Triton SSM kernels, continuing: {type(e).__name__}: {e}")
    else:
        selective_scan_triton = _selective_scan_triton
