"""Simple linear projection layer.

This is the “default” projection primitive; by keeping it tiny and
configuration-driven, you can reuse it everywhere (heads, adapters, probes)
without reintroducing bespoke glue code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn
from typing_extensions import override

from config.layer import LinearLayerConfig

try:
    from tensordict import TensorDictBase as _TensorDictBase  # type: ignore[import-not-found]
except ImportError:
    _TensorDictBase = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from tensordict import TensorDictBase  # type: ignore[import-not-found]


class LinearLayer(nn.Module):
    """Linear projection layer

    Linear layers are the main way transformer models move between feature
    spaces; most architectural variation is “where do we put projections and
    what do we do between them?”
    """

    def __init__(self, config: LinearLayerConfig) -> None:
        """Create a linear projection

        A linear layer is just an affine transform; the interesting part is not
        the math, but how many times you apply it and where you add nonlinear
        structure around it.
        """
        super().__init__()
        self.config = config
        self.linear = nn.Linear(
            config.d_in,
            config.d_out,
            bias=bool(config.bias),
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply the projection

        This is a single matrix multiply (plus optional bias), so it stays fast
        and predictable across devices.
        """
        return self.linear(x)
