"""InferContext attention mask helper for op graphs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from caramba.operation.cache.base import CacheOperation

if TYPE_CHECKING:
    from torch import Tensor


class InferCtxAttnMaskOperation(CacheOperation):
    """Read `attn_mask` from an InferContext-like object."""

    def forward(self, *, infer_ctx: object) -> "Tensor | None":  # type: ignore[override]
        if infer_ctx is None:
            return None
        # We intentionally don't require a specific ctx type; any object with
        # `.attn_mask` works (mirrors AttentionBase semantics).
        ictx = cast(Any, infer_ctx)
        return cast("Tensor | None", getattr(ictx, "attn_mask", None))

