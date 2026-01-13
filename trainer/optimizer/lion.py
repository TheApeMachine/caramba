"""Lion optimizer

The Lion optimizer is a simple optimizer that uses the Lion algorithm.
"""
from __future__ import annotations

import torch
from torch import nn

from caramba.optimizer.base import Optimizer
from caramba.manifest import Manifest
from caramba.manifest.target import Target


class Lion(Optimizer):
    def __init__(self, *, manifest: Manifest, target: Target) -> None:
        super().__init__(manifest=manifest)
        self.manifest = manifest
        self.fn = torch.optim.Lion(
            params=manifest.variables.model.parameters(), lr=manifest.variables.lr
        )

    def step(self, params: list[nn.Parameter]) -> list[nn.Parameter]:
        return torch.optim.Lion(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
