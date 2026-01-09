"""Tensor file utilities

Helper functions for detecting file formats and working with tensor files.
These utilities are shared across different tensor source implementations.
"""
from __future__ import annotations

from pathlib import Path


def is_npy(path: Path) -> bool:
    """Check if path is NumPy file

    Tests file extension to determine if a path points to a NumPy array file,
    enabling format-specific loading logic.
    """
    return path.suffix.lower() == ".npy"


def is_safetensors(path: Path) -> bool:
    """Check if path is safetensors file

    Tests file extension to identify safetensors format, which requires
    different loading logic than NumPy arrays.
    """
    return path.suffix.lower() == ".safetensors"
