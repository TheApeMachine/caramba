"""Byte-size helpers for dtypes / quantization kinds."""

from __future__ import annotations


def bytes_per_kind(kind: str) -> float:
    """Bytes per element for a precision/quantization kind string.

    Supported:
    - fp32, fp16
    - q8_0, q4_0
    """
    k = str(kind).lower()
    if k == "fp32":
        return 4.0
    if k == "fp16":
        return 2.0
    if k == "q8_0":
        return 1.0
    if k == "q4_0":
        return 0.625
    raise ValueError(f"Unknown kind: {kind}")

