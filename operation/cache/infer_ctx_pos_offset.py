"""InferContext attribute helpers for op graphs."""
from __future__ import annotations

from caramba.operation.cache.base import CacheOperation


class InferCtxPosOffsetOperation(CacheOperation):
    """Read `pos_offset` from an InferContext-like object."""

    def forward(self, *, infer_ctx: object) -> int:  # type: ignore[override]
        if infer_ctx is None:
            return 0
        return int(getattr(infer_ctx, "pos_offset", 0) or 0)

