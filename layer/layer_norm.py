"""Standard LayerNorm layer.

LayerNorm normalizes each sample independently across the feature dimension,
centering (subtracting mean) and scaling (dividing by std). While RMSNorm
is more common in modern LLMs, LayerNorm is still used in some architectures.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import LayerNormLayerConfig


class LayerNormLayer(nn.Module):
    """Layer normalization layer

    LayerNorm stabilizes optimization by keeping activations in a predictable
    range; that usually allows larger learning rates and reduces sensitivity to
    initialization.
    """

    def __init__(self, config: LayerNormLayerConfig) -> None:
        """Initialize LayerNorm

        The epsilon is a tiny constant added for numerical stability so the
        normalization never divides by zero.
        """
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(
            config.d_model,
            eps=float(config.eps),
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply LayerNorm

        Normalizing per token (across the last dimension) is a strong default
        for transformer blocks because it does not depend on sequence length.
        """
        from caramba.optimizer.kernels import layernorm

        return layernorm(
            x=x,
            weight=self.norm.weight,
            bias=self.norm.bias,
            eps=float(self.norm.eps),
        )
