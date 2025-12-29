"""High-performance 2D Convolution layer.

Wraps nn.Conv2d with Caramba's configuration interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from torch import Tensor, nn
from typing_extensions import override

from config.layer import Conv2dLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class Conv2dLayer(nn.Module):
    """2D Convolution layer."""

    def __init__(self, config: Conv2dLayerConfig) -> None:
        super().__init__()
        self.config = config
        
        # Ensure tuple formats for rigorous typing
        k = self._pair(config.kernel_size)
        s = self._pair(config.stride)
        p = self._pair(config.padding) if isinstance(config.padding, (int, tuple)) else config.padding
        d = self._pair(config.dilation)

        self.conv = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=k,
            stride=s,
            padding=p,  # type: ignore
            dilation=d,
            groups=config.groups,
            bias=config.bias,
            padding_mode=config.padding_mode,
        )

    def _pair(self, x: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        return x

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply convolution.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            ctx: Context (unused)
        """
        return self.conv(x)
