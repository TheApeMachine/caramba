"""Associative memory storage

Implements pattern storage, weight construction, and substrate network building.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthnn.core.phasor_math import PhasorMath
from synthnn.core.resonant_network import ResonantNetwork
from synthnn.core.resonant_node import ResonantNode
from synthnn.core.associative_memory.state import StoredMemory


@dataclass(frozen=True, slots=True)
class MemoryStorage:
    """Pattern storage for PhaseAssociativeMemory."""

    num_units: int
    dtype: np.dtype
    node_prefix: str
    coupling_strength: float
    damping: float
    zero_diag: bool
    math: PhasorMath

    def store(self, *, patterns: np.ndarray, labels: list[str] | None) -> StoredMemory:
        pats = self.math.as2d(np.asarray(patterns))
        if pats.shape[1] != int(self.num_units):
            raise ValueError(f"patterns must have shape (K, {self.num_units})")
        pats = self.math.toPhasors(pats, dtype=self.dtype)

        count = int(pats.shape[0])
        if labels is not None and len(labels) != count:
            raise ValueError("labels length must match number of patterns")

        weights = self.weightsFromPatterns(patterns=pats)
        net = self.networkFromWeights(weights=weights)
        return StoredMemory(patterns=pats.astype(self.dtype, copy=False), weights=weights, labels=None if labels is None else list(labels), network=net)

    def weightsFromPatterns(self, *, patterns: np.ndarray) -> np.ndarray:
        weights = (patterns.conj().T @ patterns) / float(self.num_units)
        if bool(self.zero_diag):
            np.fill_diagonal(weights, 0.0 + 0.0j)
        return weights.astype(self.dtype, copy=False)

    def networkFromWeights(self, *, weights: np.ndarray) -> ResonantNetwork:
        net = ResonantNetwork(name="phase_associative_memory")
        net.coupling_strength = float(self.coupling_strength)
        net.global_damping = float(self.damping)

        for i in range(int(self.num_units)):
            node = ResonantNode(node_id=self.nodeId(i), natural_freq=0.0, damping=float(self.damping), signal=1.0 + 0.0j)
            net.addNode(node)

        for src in range(int(self.num_units)):
            for tgt in range(int(self.num_units)):
                if bool(self.zero_diag) and src == tgt:
                    continue
                w = complex(weights[tgt, src])
                if w == 0.0 + 0.0j:
                    continue
                net.connect(source_id=self.nodeId(src), target_id=self.nodeId(tgt), weight=w, delay=0.0)
        return net

    def nodeId(self, index: int) -> str:
        return f"{self.node_prefix}{int(index)}"

