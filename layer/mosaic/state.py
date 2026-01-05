"""State management for MOSAIC block.

Provides the streaming state container and a context-backed store for keeping
per-layer state across decode steps without global variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass
class MosaicState:
    """MOSAIC streaming state.

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


class MosaicStateStore:
    """Context-backed state store.

    Uses `ctx._mosaic` as a dict keyed by per-layer ids. This keeps streaming state
    attached to the caller-provided context object and avoids hidden globals.
    """

    class Ctx(Protocol):
        _mosaic: dict[str, MosaicState]

    def get(self, ctx: Ctx | None, *, key: str) -> MosaicState | None:
        if ctx is None:
            return None
        store = getattr(ctx, "_mosaic", None)
        if not isinstance(store, dict):
            return None
        st = store.get(str(key))
        return st if isinstance(st, MosaicState) else None

    def set(self, ctx: Ctx | None, *, key: str, state: MosaicState) -> None:
        if ctx is None:
            return
        store = getattr(ctx, "_mosaic", None)
        if store is None:
            store = {}
            try:
                setattr(ctx, "_mosaic", store)  # type: ignore[arg-type]
            except Exception as e:
                raise RuntimeError(f"Failed to set ctx._mosaic state store: {e}") from e
        if not isinstance(store, dict):
            raise TypeError(
                f"ctx._mosaic exists but is not a dict: {type(store).__name__}. "
                "Expected dict[str, MosaicState] for state storage."
            )
        store[str(key)] = state
