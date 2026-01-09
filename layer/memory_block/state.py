"""Streaming memory block state management

Streaming models need persistent state across decode steps; this module keeps
that state explicit and context-scoped so generation stays reproducible and does
not rely on hidden globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass
class MemoryBlockState:
    """Streaming memory block state.

    Holds fixed-size buffers used across decode steps: local conv buffer, multiscale
    state bank, optional registers, and set-associative memory tables.
    """

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
    # VSA tags per slot for robust in-bucket selection: (B, H, buckets, assoc, vsa_dim)
    mem_tag: Tensor
    mem_last: Tensor

    # Optional Resonant Memory Field (RMF) state.
    # Stored as (B, rmf_dim, 2) where the last dim is (real, imag).
    rmf_field: Tensor | None = None


class MemoryBlockStateStore:
    """Context-backed state store

    Storing state on the caller-provided context object makes state ownership
    obvious: whoever owns the context owns the state, which is especially handy
    when debugging multi-request inference servers.
    """

    class Ctx(Protocol):
        _memblock: dict[str, MemoryBlockState]

    def get(self, ctx: Ctx | None, *, key: str) -> MemoryBlockState | None:
        if ctx is None:
            return None
        store = getattr(ctx, "_memblock", None)
        if not isinstance(store, dict):
            return None
        st = store.get(str(key))
        return st if isinstance(st, MemoryBlockState) else None

    def set(self, ctx: Ctx | None, *, key: str, state: MemoryBlockState) -> None:
        if ctx is None:
            return
        store = getattr(ctx, "_memblock", None)
        if store is None:
            store = {}
            try:
                setattr(ctx, "_memblock", store)  # type: ignore[arg-type]
            except Exception as e:
                raise RuntimeError(f"Failed to set ctx._memblock state store: {e}") from e
        if not isinstance(store, dict):
            raise TypeError(
                f"ctx._memblock exists but is not a dict: {type(store).__name__}. "
                "Expected dict[str, MemoryBlockState] for state storage."
            )
        store[str(key)] = state
