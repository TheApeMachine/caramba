"""Backend availability detection (Triton + Metal/MPS).

Caramba has optional fused kernels that depend on accelerator-specific toolchains:
- Triton (CUDA) for fused decode / SSM kernels
- Metal (MPS) for Apple Silicon fused DBA decode (custom MSL kernel + ObjC++ bridge)

This module centralizes runtime detection in a way that is safe for import + type
checking: at type-check time we force optional backends off.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from typing import TYPE_CHECKING

__all__ = [
    "TRITON_AVAILABLE",
    "METAL_SUPPORTED",
    "METAL_BUILD_TOOLS_AVAILABLE",
    "triton_decoupled_q4q8q4_available",
    "triton_ssm_available",
    "metal_supported",
    "metal_build_tools_available",
]


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


# At type-check time, force these off so optional code stays behind guards.
TRITON_AVAILABLE: bool = (
    False
    if TYPE_CHECKING
    else bool(_has_module("triton") and _has_module("triton.language"))
)


def triton_decoupled_q4q8q4_available() -> bool:
    """Check if fused decoupled q4/q8/q4 decode kernels can be used."""
    return bool(TRITON_AVAILABLE)


def triton_ssm_available() -> bool:
    """Check if fused SSM kernels can be used."""
    return bool(TRITON_AVAILABLE)


def metal_supported() -> bool:
    """Whether the current runtime *can* execute custom Metal (MPS) ops.

    This indicates platform + PyTorch MPS support. It does NOT guarantee that the
    custom extension is already built/loaded; higher-level code may JIT build it.
    """
    if TYPE_CHECKING:
        return False
    if platform.system() != "Darwin":
        return False
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def metal_build_tools_available() -> bool:
    """Whether the host can compile Metal shaders via Xcode toolchain.

    Notes:
    - Having `xcrun` in PATH is not sufficient; the active developer directory
      must contain the `metal` and `metallib` tools.
    - This function is intentionally conservative: if we can't *prove* the tools
      exist, we return False so training can surface a clear, actionable error.
    """
    if not metal_supported():
        return False
    if shutil.which("xcrun") is None:
        return False
    # Ensure the actual Metal tools exist in the selected toolchain.
    try:
        subprocess.check_output(["xcrun", "-sdk", "macosx", "--find", "metal"], stderr=subprocess.STDOUT)
        subprocess.check_output(["xcrun", "-sdk", "macosx", "--find", "metallib"], stderr=subprocess.STDOUT)
    except Exception:
        return False
    return True


METAL_SUPPORTED: bool = bool(metal_supported()) if not TYPE_CHECKING else False
METAL_BUILD_TOOLS_AVAILABLE: bool = (
    bool(metal_build_tools_available()) if not TYPE_CHECKING else False
)

