"""Append to a Caramba KV cache.

This operation appends one step (or a chunk) of new cached tensors to a cache
object. The cache object can be:
- LayerKVCache (k,v)
- DecoupledLayerKVCache (k_sem,k_geo,v)
- MultiKVCache (arbitrary named fields)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from caramba.operation.cache.base import CacheOperation

if TYPE_CHECKING:
    from collections.abc import Sequence


class KVCacheWriteOperation(CacheOperation):
    """Append new tokens to a cache object."""

    def __init__(self, *, keys: "Sequence[str] | None" = None) -> None:
        super().__init__()
        self.keys = tuple(keys) if keys is not None else None

    def forward(self, cache: object, *xs: Tensor) -> tuple[object, int]:  # type: ignore[override]
        keys = self.keys
        if keys is None:
            keys = tuple(getattr(cache, "keys", ()))
        if not keys:
            raise ValueError("KVCacheWriteOperation requires non-empty keys (pass keys=... or cache.keys)")
        if len(xs) != len(keys):
            raise ValueError(f"KVCacheWriteOperation expected {len(keys)} tensors, got {len(xs)}")
        if not hasattr(cache, "append_many"):
            raise TypeError("cache must implement append_many(...)")
        items = {k: x for k, x in zip(keys, xs, strict=True)}
        old = int(cache.append_many(items))  # type: ignore[attr-defined]
        # Return the cache as a passthrough so graphs can enforce ordering by
        # wiring cache -> cache between nodes.
        return cache, old
