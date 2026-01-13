"""Cache position helpers for op graphs."""
from __future__ import annotations

from caramba.operation.cache.base import CacheOperation


class KVCachePosOperation(CacheOperation):
    """Read `.pos` from a cache object."""

    def forward(self, cache: object) -> int:  # type: ignore[override]
        if cache is None:
            return 0
        return int(getattr(cache, "pos", 0) or 0)

