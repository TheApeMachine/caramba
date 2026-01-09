"""Graph NumPy dataset

Loads a single graph from NumPy arrays containing node features, adjacency
matrix, and labels. Designed for graph-level tasks where the entire graph is
a single sample, avoiding the need for external graph processing libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch

from caramba.runtime.tensordict_utils import TensorDictBase
from caramba.data.graphs.single import SingleGraphDataset


@dataclass(frozen=True, slots=True)
class GraphNpyDataset:
    """Graph NumPy dataset component

    Manifest-level dataset that loads node features, adjacency matrix, and
    labels from separate NumPy files. Optionally adds self-loops to the
    adjacency matrix for graph neural network training.
    """
    x_path: str
    adj_path: str
    labels_path: str
    add_self_loops: bool = True

    def build(self) -> Dataset[TensorDictBase]:
        """Build graph dataset

        Loads all three NumPy files, converts them to tensors, and optionally
        adds self-loops to the adjacency matrix (diagonal elements set to 1),
        which is a common preprocessing step for GNNs.
        """
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

        return SingleGraphDataset(x=x_t, adj=adj_t, labels=labels_t)

