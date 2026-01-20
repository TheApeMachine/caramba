"""Dropout layer

Dropout regularizes by making a network robust to missing activations; at
training time it behaves like an ensemble of many thinned networks, which often
improves generalization.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from config.layer import DropoutLayerConfig


class DropoutLayer(nn.Module):
    """Dropout layer

    The key trick is that dropout is only meaningful during training; at eval
    time the layer becomes an identity so your model is deterministic.
    """

    def __init__(self, config: DropoutLayerConfig) -> None:
        """Initialize dropout

        The probability p controls how aggressively activations are masked;
        higher values force the model to spread information across features.
        """
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.p)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply dropout

        PyTorch handles the train/eval toggle internally, so this method stays a
        simple pass-through wrapper.
        """
        return self.dropout(x)
