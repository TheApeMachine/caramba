"""AdamW optimizer

An improvement over the standard Adam, that fixes how weight decay (L2 regularization)
is applied, leading to better generalization and performance by decoupling it from the
gradient updates, applying it directly to the weights, and making it more effective,
especially in models like transformers. It combines Adam's adaptive learning rates with
proper weight decay, making it a go-to choice for training complex neural networks.
"""
from __future__ import annotations

import torch
from torch import nn

from caramba.optimizer.base import Optimizer
from caramba.config.manifest import Manifest


class AdamW(Optimizer):
    def __init__(self, *, manifest: Manifest):
        super().__init__(manifest=manifest)
        self.manifest = manifest
        self.fn = torch.optim.AdamW(
            params=manifest.params, lr=manifest.lr
        )

    def step(self, params: list[nn.Parameter]) -> list[nn.Parameter]:
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        optimizer.step()
