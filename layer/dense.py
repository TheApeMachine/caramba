"""Dense block layer

Dense “blocks” (linear + optional norm/activation/dropout) are the workhorse of
MLPs; packaging them as one layer makes it easy to express common patterns
without repeating boilerplate across architectures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import DenseLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class DenseLayer(nn.Module):
    """Dense block layer

    In practice, the exact ordering of projection/norm/activation is an
    architectural choice; keeping it in one configurable block makes those
    choices cheap to explore.
    """

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
            from caramba.layer.rms_norm import RMSNormLayer
            from caramba.config.layer import RMSNormLayerConfig
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
        """Apply dense block

        This is the classic “feature transform” step: project to a new space,
        optionally normalize, apply a nonlinearity, then regularize.
        """
        x = self.linear(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        x = self.dropout(x)
        return x
