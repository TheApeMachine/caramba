"""Apply attention mask by adding negative infinity

Prevents attention to masked positions by setting their scores to -inf.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.masking.base import MaskingOperation


class ApplyMaskOperation(MaskingOperation):
    """Apply attention mask

    Applies a boolean mask to attention scores by setting masked positions to -inf,
    ensuring softmax will give zero probability to masked positions.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *, scores: Tensor, mask: Tensor) -> Tensor:
        """Apply mask to attention scores

        Sets attention scores at masked positions to negative infinity,
        ensuring they receive zero probability after softmax normalization.
        """
        # mask should be True for positions to mask (not attend to)
        return scores.masked_fill(mask, float('-inf'))