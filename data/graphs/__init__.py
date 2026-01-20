"""Graph datasets package

Loads a single graph from NumPy arrays containing node features, adjacency
matrix, and labels. Designed for graph-level tasks where the entire graph is
a single sample, avoiding the need for external graph processing libraries.
"""
from __future__ import annotations

from data.graphs.npy import GraphNpyDataset
from data.graphs.single import SingleGraphDataset


__all__ = ["GraphNpyDataset", "SingleGraphDataset"]