"""Key-value cache operation base classes

Foundation classes for KV-cache operations in autoregressive generation.
"""
from __future__ import annotations

from caramba.operation.base import Operation


class CacheOperation(Operation):
    """Base class for all cache operations

    Provides common interface for operations that manage key-value caches
    used in efficient autoregressive generation and sequence processing.
    """
    def __init__(self) -> None:
        super().__init__()