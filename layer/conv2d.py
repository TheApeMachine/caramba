"""2D convolution layer

Convolutions are still a great inductive bias for local structure; wrapping
`nn.Conv2d` as a manifest-friendly layer makes it easy to mix CNN components
into otherwise transformer-centric experiments.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import Conv2dLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class Conv2dLayer(nn.Module):
    """2D convolution layer

    A small wrapper like this is mostly about ergonomics: it keeps construction
    consistent with the rest of the platform (config in, module out).
    """

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
        """Apply convolution

        Convolution is a learned, translation-equivariant filter; it is often a
        good fit when you want spatial locality “for free” instead of learning
        it purely through attention.
        """
        return self.conv(x)
