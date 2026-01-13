"""SGD optimizer

The SGD optimizer is a simple optimizer that uses the stochastic gradient descent algorithm.
"""
from __future__ import annotations

import torch
from torch import nn

from caramba.optimizer.base import Optimizer
from caramba.manifest import Manifest
from caramba.manifest.target import Target


class SGD(Optimizer):
    def __init__(self, *, manifest: Manifest, target: Target) -> None:
        super().__init__(manifest=manifest, target=target)
        self.manifest = manifest
        self.target = target
        self.fn = torch.optim.SGD(
            params=manifest.params, lr=manifest.lr
        )

    def step(self, params: list[nn.Parameter]) -> list[nn.Parameter]:
        optimizer = torch.optim.SGD(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
