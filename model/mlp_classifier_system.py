"""A simple MLP classifier system.

This is a deliberately small, non-language-model baseline to demonstrate that
Caramba's manifest-driven pipeline is not scoped to LMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class MLPClassifierSystem:
    """MLP classifier.

    Config:
    - d_in: input feature dimension
    - d_hidden: hidden dimension
    - n_layers: number of hidden layers
    - n_classes: output classes
    - dropout: dropout probability (optional)
    """

    d_in: int
    d_hidden: int
    n_layers: int
    n_classes: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        layers: list[nn.Module] = []
        din = int(self.d_in)
        for _ in range(int(self.n_layers)):
            layers.append(nn.Linear(din, int(self.d_hidden)))
            layers.append(nn.ReLU())
            if float(self.dropout) > 0:
                layers.append(nn.Dropout(p=float(self.dropout)))
            din = int(self.d_hidden)
        layers.append(nn.Linear(din, int(self.n_classes)))
        self.module = nn.Sequential(*layers)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "MLPClassifierSystem":
        self.module = self.module.to(device=device, dtype=dtype)
        return self

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        _ = ctx
        x: Tensor = batch["inputs"]
        logits = self.module(x)
        return {"logits": logits}

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

