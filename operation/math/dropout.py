"""Dropout regularization operation

Randomly zeros out elements during training for regularization.
"""
from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from caramba.operation.math.base import MathOperation


class DropoutOperation(MathOperation):
    """Dropout regularization

    Randomly zeros out elements of input tensor during training with probability p,
    helping prevent overfitting by forcing the network to learn redundant representations.
    """
    def __init__(self, *, p: float = 0.1, training: bool = True) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("dropout probability must be between 0 and 1")
        self.p = p
        self.training = training

    def forward(self, *, x: Tensor) -> Tensor:
        """Apply dropout regularization

        During training, randomly zeros out elements with probability p.
        During inference, returns input unchanged (but scaled appropriately).
        """
        return F.dropout(x, p=self.p, training=self.training)