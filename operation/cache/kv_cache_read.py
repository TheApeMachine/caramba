"""Read from a Caramba KV cache.

This operation reads cached tensors from a per-layer cache object (standard,
decoupled, or any future variant that implements the small `*many` API).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from caramba.operation.cache.base import CacheOperation

if TYPE_CHECKING:
    from collections.abc import Sequence


class KVCacheReadOperation(CacheOperation):
    """Read cached token ranges for one cache object."""

    def __init__(self, *, keys: "Sequence[str] | None" = None) -> None:
        super().__init__()
        self.keys = tuple(keys) if keys is not None else None

    def forward(  # type: ignore[override]
        self,
        cache: object,
        *,
        start_pos: int = 0,
        seq_len: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> Tensor | tuple[Tensor, ...]:
        keys = self.keys
        if keys is None:
            keys = tuple(getattr(cache, "keys", ()))
        if not keys:
            raise ValueError("KVCacheReadOperation requires non-empty keys (pass keys=... or cache.keys)")

        pos = int(getattr(cache, "pos", 0))
        start = int(start_pos)
        end = int(pos) if seq_len is None else int(start + int(seq_len))
        if start < 0 or end < start:
            raise ValueError(f"Invalid cache slice {start}:{end}")
        if end > pos:
            raise ValueError(f"Requested end {end} > cache.pos {pos}")

        if hasattr(cache, "get_slice_many"):
            out = cache.get_slice_many(start, end, dtype=dtype)  # type: ignore[attr-defined]
        elif hasattr(cache, "get_many"):
            all_items = cache.get_many(dtype=dtype)  # type: ignore[attr-defined]
            out = {k: all_items[k][:, start:end] for k in keys}
        else:
            raise TypeError("cache must implement get_slice_many(...) or get_many(...)")

        if len(keys) == 1:
            return out[keys[0]]
        return tuple(out[k] for k in keys)
