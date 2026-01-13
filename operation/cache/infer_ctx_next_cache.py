"""Operations for interacting with InferContext from op graphs."""
from __future__ import annotations

from caramba.operation.cache.base import CacheOperation


class InferCtxNextCacheOperation(CacheOperation):
    """Return the next per-layer cache from an InferContext-like object."""

    def forward(self, *, infer_ctx: object) -> object:  # type: ignore[override]
        if infer_ctx is None:
            raise ValueError("infer_ctx is required")
        next_cache = getattr(infer_ctx, "next_cache", None)
        if next_cache is None or not callable(next_cache):
            raise TypeError("infer_ctx must provide a callable next_cache()")
        return next_cache()

