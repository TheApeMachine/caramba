"""Combine multiple attention masks

Merges different types of masks using logical operations.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.masking.base import MaskingOperation


class CombineMasksOperation(MaskingOperation):
    """Combine multiple attention masks

    Merges different attention masks using logical OR operation,
    useful when combining causal masks with padding masks or other constraints.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, masks: list[Tensor]) -> Tensor:
        """Combine masks with logical OR

        Combines multiple boolean masks by taking their logical OR,
        creating a final mask where any masked position from any input mask is masked.
        """
        if not masks:
            raise ValueError("At least one mask must be provided")

        combined = masks[0]
        for mask in masks[1:]:
            combined = combined | mask
        return combined