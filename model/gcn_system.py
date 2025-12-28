"""A minimal GCN system for node classification.

This is intentionally lightweight (no PyG/DGL dependency) and meant to show the
platform is not LM-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def _normalize_adj(adj: Tensor) -> Tensor:
    # adj: (N, N) dense with self-loops
    deg = torch.sum(adj, dim=-1)  # (N,)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    d = torch.diag(deg_inv_sqrt)
    return d @ adj @ d


@dataclass
class GCNSystem:
    """2-layer GCN node classifier.

    Config:
    - d_in: input features
    - d_hidden: hidden dimension
    - n_classes: output classes
    - dropout: dropout probability
    """

    d_in: int
    d_hidden: int
    n_classes: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        self.lin1 = nn.Linear(int(self.d_in), int(self.d_hidden), bias=False)
        self.lin2 = nn.Linear(int(self.d_hidden), int(self.n_classes), bias=False)
        self.drop = nn.Dropout(p=float(self.dropout)) if float(self.dropout) > 0 else nn.Identity()

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "GCNSystem":
        for m in (self.lin1, self.lin2, self.drop):
            m.to(device=device, dtype=dtype)
        return self

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        _ = ctx
        x: Tensor = batch["x"]          # (N, F)
        adj: Tensor = batch["adj"]      # (N, N)
        a = _normalize_adj(adj)
        h = a @ self.lin1(x)
        h = torch.relu(h)
        h = self.drop(h)
        logits = a @ self.lin2(h)
        return {"logits": logits}

    def parameters(self):
        return list(self.lin1.parameters()) + list(self.lin2.parameters())

    def state_dict(self):
        return {"lin1": self.lin1.state_dict(), "lin2": self.lin2.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lin1.load_state_dict(state["lin1"])
        self.lin2.load_state_dict(state["lin2"])

