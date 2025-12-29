"""High-performance Dense (Fully Connected) layer.

Combines Linear projection with optional Normalization, Activation, and Dropout
into a single fused block for efficiency and convenience.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from torch import Tensor, nn
from typing_extensions import override

from config.layer import DenseLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class DenseLayer(nn.Module):
    """Fused Dense Layer (Linear -> Norm -> Act -> Dropout)."""

    def __init__(self, config: DenseLayerConfig) -> None:
        super().__init__()
        self.config = config

        self.linear = nn.Linear(
            config.d_in,
            config.d_out,
            bias=config.bias,
        )

        self.norm: nn.Module | None = None
        if config.normalization == "layer_norm":
            self.norm = nn.LayerNorm(config.d_out)
        elif config.normalization == "rms_norm":
            # Avoid circular import if possible, or use simple implementation
            from layer.rms_norm import RMSNormLayer
            from config.layer import RMSNormLayerConfig
            self.norm = RMSNormLayer(RMSNormLayerConfig(d_model=config.d_out))

        self.act: nn.Module | None = None
        if config.activation:
            if config.activation == "relu":
                self.act = nn.ReLU()
            elif config.activation == "gelu":
                self.act = nn.GELU()
            elif config.activation == "silu":
                self.act = nn.SiLU()
            elif config.activation == "tanh":
                self.act = nn.Tanh()
            else:
                # Fallback to getting from torch.nn
                try:
                    self.act = getattr(nn, config.activation)()
                except AttributeError:
                    raise ValueError(f"Unknown activation: {config.activation}")

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply dense layer."""
        x = self.linear(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        x = self.dropout(x)
        return x
