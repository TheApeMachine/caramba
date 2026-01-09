"""Single graph dataset

Wraps a complete graph (nodes, edges, labels) as a single dataset sample,
making it easy to work with graph-level prediction tasks.
"""
from __future__ import annotations

from torch.utils.data import Dataset
from caramba.runtime.tensordict_utils import TensorDictBase
from torch import Tensor

from caramba.runtime.tensordict_utils import as_tensordict


class SingleGraphDataset(Dataset[TensorDictBase]):
    """Single graph dataset implementation

    Wraps a complete graph (nodes, edges, labels) as a single dataset sample,
    making it easy to work with graph-level prediction tasks.
    """
    def __init__(self, *, x: Tensor, adj: Tensor, labels: Tensor) -> None:
        """Initialize single graph dataset

        Stores the graph components as a single item, since the dataset
        contains only one graph sample.
        """
        self.item = {"x": x, "adj": adj, "labels": labels}

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
        return as_tensordict(self.item)