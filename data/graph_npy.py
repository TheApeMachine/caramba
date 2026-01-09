"""Graph NumPy dataset

Loads a single graph from NumPy arrays containing node features, adjacency
matrix, and labels. Designed for graph-level tasks where the entire graph is
a single sample, avoiding the need for external graph processing libraries.
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
    """Single graph dataset implementation

    Wraps a complete graph (nodes, edges, labels) as a single dataset sample,
    making it easy to work with graph-level prediction tasks.
    """
    def __init__(self, *, x: Tensor, adj: Tensor, labels: Tensor) -> None:
        """Initialize single graph dataset

        Stores the graph components as a single item, since the dataset
        contains only one graph sample.
        """
        self._item = {"x": x, "adj": adj, "labels": labels}

    def __len__(self) -> int:
        """Get dataset length

        Returns 1 since this dataset contains a single graph, making it
        compatible with DataLoader interfaces while representing one sample.
        """
        return 1

    def __getitem__(self, idx: int) -> TensorDictBase:
        """Get graph sample

        Returns the complete graph regardless of index, since there's only
        one graph in the dataset. The index parameter is ignored but required
        by the Dataset protocol.
        """
        _ = idx
        return as_tensordict(self._item)


