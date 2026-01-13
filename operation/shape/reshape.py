"""Tensor reshaping operation

Changes the shape of a tensor while preserving total number of elements.
"""
from __future__ import annotations

from torch import Tensor

from caramba.operation.base import Operation


class ReshapeOperation(Operation):
    """Change tensor dimensions

    Rearranges tensor elements into a new shape with the same total number of elements,
    essential for adapting tensors between different neural network layer requirements.
    """
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, *, x: Tensor) -> Tensor:
        """Reshape tensor to new dimensions

        Rearranges tensor elements into the specified shape, maintaining the same
        total number of elements but changing how they're organized across dimensions.
        """
        return x.reshape(self.shape)