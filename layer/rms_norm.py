"""RMSNorm: a simpler alternative to LayerNorm.

RMSNorm normalizes by the root mean square instead of mean and variance.
It's computationally cheaper and empirically works just as well for LLMs.
Llama and many modern models use RMSNorm instead of LayerNorm.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import RMSNormLayerConfig


class RMSNormLayer(nn.Module):
    """Root mean square normalization layer

    RMSNorm skips mean-centering and only rescales by the RMS, which often
    preserves the stability benefits of normalization while reducing compute and
    parameter count.
    """

    def __init__(self, config: RMSNormLayerConfig) -> None:
        """Initialize RMSNorm

        The learnable weight gives the model a way to “undo” normalization where
        helpful, while still keeping activations numerically well-behaved.
        """
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.eps = float(config.eps)
        self.elementwise_affine = bool(config.elementwise_affine)
        self.weight: nn.Parameter | None = (
            nn.Parameter(torch.ones(self.d_model))
            if self.elementwise_affine
            else None
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Normalize by root mean square

        RMSNorm rescales each token by a single scalar derived from its feature
        magnitudes, which is a cheap way to keep layer activations in a stable
        range.
        """
        if x.ndim < 1:
            raise ValueError(f"Expected x.ndim >= 1, got {x.shape}")
        if int(x.shape[-1]) != int(self.d_model):
            raise ValueError(f"Expected x last dim {int(self.d_model)}, got {x.shape}")

        from caramba.optimizer.kernels import rmsnorm

        return rmsnorm(x=x, weight=self.weight, eps=float(self.eps))
