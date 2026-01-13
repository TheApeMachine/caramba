"""Key-value cache operations

Operations for managing KV-cache in autoregressive generation,
enabling efficient decoding by reusing previously computed keys and values.
"""
from __future__ import annotations

from caramba.operation.cache.base import CacheOperation
from caramba.operation.cache.cache_pos import KVCachePosOperation
from caramba.operation.cache.infer_ctx_attn_mask import InferCtxAttnMaskOperation
from caramba.operation.cache.infer_ctx_next_cache import InferCtxNextCacheOperation
from caramba.operation.cache.infer_ctx_pos_offset import InferCtxPosOffsetOperation
from caramba.operation.cache.kv_cache_read import KVCacheReadOperation
from caramba.operation.cache.kv_cache_write import KVCacheWriteOperation

__all__ = [
    "CacheOperation",
    "InferCtxAttnMaskOperation",
    "InferCtxNextCacheOperation",
    "InferCtxPosOffsetOperation",
    "KVCachePosOperation",
    "KVCacheReadOperation",
    "KVCacheWriteOperation",
]
