"""Masking operation base classes

Foundation classes for attention masking operations in neural networks.
"""
from __future__ import annotations

from caramba.operation.base import Operation


class MaskingOperation(Operation):
    """Base class for all masking operations

    Provides common interface for operations that create and apply masks
    for attention mechanisms and sequence processing.
    """
    def __init__(self) -> None:
        super().__init__()