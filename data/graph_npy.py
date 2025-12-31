"""Single-graph dataset loaded from NumPy arrays.

This provides a minimal bridge for graph research without requiring external
graph libraries. It yields a single item containing the full graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict

class _SingleGraphDataset(Dataset[TensorDictBase]):
    def __init__(self, *, x: Tensor, adj: Tensor, labels: Tensor) -> None:
        self._item = {"x": x, "adj": adj, "labels": labels}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> TensorDictBase:
        _ = idx
        return as_tensordict(self._item)


@dataclass(frozen=True, slots=True)
class GraphNpyDataset:
    """Graph dataset component.

    Expected files:
    - x_path: node features, shape (N, F)
    - adj_path: adjacency matrix, shape (N, N) (dense) with 0/1 weights
    - labels_path: node labels, shape (N,)
    """

    x_path: str
    adj_path: str
    labels_path: str
    add_self_loops: bool = True

    def build(self) -> Dataset[TensorDictBase]:
        x = np.load(Path(self.x_path))
        adj = np.load(Path(self.adj_path))
        labels = np.load(Path(self.labels_path))

        x_t = torch.from_numpy(x).float()
        adj_t = torch.from_numpy(adj).float()
        labels_t = torch.from_numpy(labels).long()

        if self.add_self_loops:
            n = adj_t.shape[0]
            adj_t = adj_t.clone()
            adj_t += torch.eye(n, dtype=adj_t.dtype)

        return _SingleGraphDataset(x=x_t, adj=adj_t, labels=labels_t)

