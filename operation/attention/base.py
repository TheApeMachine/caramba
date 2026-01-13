"""Attention mechanism operation base classes

Foundation classes for attention operations in neural networks.
"""
from __future__ import annotations

from caramba.operation.base import Operation


class AttentionOperation(Operation):
    """Base class for all attention operations

    Provides common interface for attention mechanisms that compute
    weighted combinations of values based on query-key similarities.
    """
    def __init__(self) -> None:
        super().__init__()