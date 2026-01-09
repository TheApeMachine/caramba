"""Streaming memory block components.

This package contains composable building blocks for *streaming* models that use
explicit state (local buffers, multiscale integrators, and fixed-size addressed
memory) rather than attention/KV-caches.
"""

from caramba.layer.memory_block.block import MemoryBlockLayer
from caramba.layer.memory_block.ngram_cache import NGramCacheLogitsLayer

__all__ = [
    "MemoryBlockLayer",
    "NGramCacheLogitsLayer",
]

