"""Precision / dtype selection helpers."""

from __future__ import annotations

import torch


def neg_inf(dtype: torch.dtype) -> float:
    """Return a large negative value safe for the given dtype.

    Notes:
    - For fp16 we avoid values outside representable range (e.g. -1e9), which
      can overflow to -inf and sometimes interact poorly with fused kernels.
    - For other dtypes we use a conventional large negative mask value.
    """
    if dtype == torch.float16:
        return -65504.0
    return -1e9


def autocast_dtype(device: torch.device, spec: str) -> torch.dtype:
    """Resolve autocast dtype from a user spec (including 'auto')."""
    s = str(spec).lower()
    if s == "auto":
        if device.type == "cuda":
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.bfloat16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float16


def autocast_dtype_str(device: torch.device) -> str:
    """Default autocast dtype string for 'auto'."""
    if device.type == "cuda":
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bfloat16"
        except Exception:
            pass
        return "float16"
    if device.type == "mps":
        return "float16"
    return "bfloat16"


def weight_dtype(device: torch.device, spec: str) -> torch.dtype:
    """Resolve model *weight* dtype from a user spec (including 'auto')."""
    s = str(spec).lower()
    if s == "auto":
        if device.type == "cuda":
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float16":
        return torch.float16
    return torch.float32


def weight_dtype_str(device: torch.device) -> str:
    """Default weight dtype string for 'auto'."""
    if device.type == "cuda":
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bfloat16"
        except Exception:
            pass
        return "float16"
    if device.type == "mps":
        return "float16"
    return "float32"

