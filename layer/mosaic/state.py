"""State management for MOSAIC block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class MosaicState:
    # conv buffer holds the last (k-1) normalized vectors: (B, k-1, d_model)
    conv_buf: Tensor
    # multiscale state bank: (B, K, d_model)
    s: Tensor
    # non-decaying register file (optional): (B, R, d_model)
    regs: Tensor | None
    # global step counter (for LRU replacement via per-slot timestamps)
    step: int
    # set-associative hash memory:
    # - keys: (B, H, buckets, assoc, key_dim)
    # - vals: (B, H, buckets, assoc, mem_dim)
    # - last: (B, H, buckets, assoc) int64 timestamps, -1 means empty
    mem_k: Tensor
    mem_v: Tensor
    mem_last: Tensor


def get_state(ctx: object | None, key: str) -> MosaicState | None:
    if ctx is None:
        return None
    store = getattr(ctx, "_mosaic", None)
    if store is None:
        return None
    if not isinstance(store, dict):
        return None
    st = store.get(key)
    return st if isinstance(st, MosaicState) else None


def set_state(ctx: object | None, key: str, st: MosaicState) -> None:
    if ctx is None:
        return
    store = getattr(ctx, "_mosaic", None)
    if store is None:
        store = {}
        try:
            setattr(ctx, "_mosaic", store)  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError(f"Failed to set state: {e}") from e
    if not isinstance(store, dict):
        raise RuntimeError(
            f"ctx._mosaic exists but is not a dict: {type(store).__name__}. "
            f"Expected dict for state storage."
        )
    store[key] = st
