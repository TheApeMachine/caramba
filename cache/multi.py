"""Generic multi-tensor KV cache for attention variants.

Caramba historically exposed two cache shapes:
- LayerKVCache: (k, v)
- DecoupledLayerKVCache: (k_sem, k_geo, v)

For manifest-defined attention graphs, we want cache *shape* to be purely
configurable so new attention variants don't require new cache classes.

MultiKVCache is a small, named bundle of SeqCacheTensor objects that all share
the same sequence position. It supports the same quantized storage options as
the built-in caches via SeqCacheTensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from caramba.cache.tensor import SeqCacheTensor
from caramba.config.kvcache import KVCacheTensorConfig

if TYPE_CHECKING:
    from collections.abc import Mapping
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class CacheFieldSpec:
    """Specification for one cached tensor stream."""

    name: str
    dim: int
    cfg: KVCacheTensorConfig


class MultiKVCache:
    """A named bundle of cache tensors for one layer/op.

    The cache stores one SeqCacheTensor per field. All fields are kept
    synchronized so `pos` is shared.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        fields: list[CacheFieldSpec],
        device: torch.device,
    ) -> None:
        if not fields:
            raise ValueError("MultiKVCache requires at least one field")
        names = [str(f.name) for f in fields]
        if any(not n for n in names):
            raise ValueError("MultiKVCache field names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"MultiKVCache field names must be unique, got {names}")

        self._order: tuple[str, ...] = tuple(names)
        self._tensors: dict[str, SeqCacheTensor] = {}
        for f in fields:
            self._tensors[str(f.name)] = SeqCacheTensor(
                batch_size=int(batch_size),
                max_seq_len=int(max_seq_len),
                dim=int(f.dim),
                cfg=f.cfg,
                device=device,
            )

    @property
    def keys(self) -> tuple[str, ...]:
        """Stable field ordering for this cache."""

        return self._order

    @property
    def pos(self) -> int:
        """Current sequence position (tokens cached so far)."""

        first = self._tensors[self._order[0]]
        p0 = int(first.pos)
        for name in self._order[1:]:
            if int(self._tensors[name].pos) != p0:
                raise RuntimeError("MultiKVCache desynced: fields have different pos")
        return p0

    def append_many(self, items: "Mapping[str, Tensor]") -> int:
        """Append new tokens to all fields.

        items must contain exactly the cache's keys (no missing/extra fields).
        """

        got = set(map(str, items.keys()))
        want = set(self._order)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise KeyError(f"MultiKVCache append_many mismatch (missing={missing}, extra={extra})")

        pos0: int | None = None
        for name in self._order:
            pos = int(self._tensors[name].append(items[name]))
            if pos0 is None:
                pos0 = pos
            elif pos != pos0:
                raise RuntimeError("MultiKVCache append position mismatch")
        if pos0 is None:
            raise RuntimeError("MultiKVCache append_many: internal error (no fields)")
        return int(pos0)

    def get_many(self, *, dtype: torch.dtype = torch.float16) -> dict[str, "Tensor"]:
        """Retrieve all cached tokens for all fields."""

        return {name: self._tensors[name].get(dtype=dtype) for name in self._order}

    def get_slice_many(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, "Tensor"]:
        """Retrieve a slice of cached tokens for all fields."""

        return {
            name: self._tensors[name].get_slice(int(start), int(end), dtype=dtype)
            for name in self._order
        }

    def truncate(self, new_pos: int) -> None:
        """Rollback all fields to a previous position."""

        new_pos = int(new_pos)
        for name in self._order:
            self._tensors[name].truncate(new_pos)
        _ = self.pos  # Validate synchronization
