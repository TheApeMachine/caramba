"""Associative memory storage

Implements pattern storage, weight construction, and substrate network building.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resonant.core.phasor_math import PhasorMath
from resonant.core.resonant_network import ResonantNetwork
from resonant.core.resonant_node import ResonantNode
from resonant.core.associative_memory.state import StoredMemory


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
    initial_signal: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        num_units = int(self.num_units)
        if num_units <= 0:
            raise ValueError(f"num_units must be > 0, got {num_units}")
        dtype = np.dtype(self.dtype)
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError(f"dtype must be complex, got {dtype}")
        coupling_strength = float(self.coupling_strength)
        damping = float(self.damping)
        if not np.isfinite(coupling_strength) or coupling_strength < 0.0:
            raise ValueError(f"coupling_strength must be finite and >= 0, got {coupling_strength}")
        if not np.isfinite(damping) or damping < 0.0 or damping > 1.0:
            raise ValueError(f"damping must be finite and in [0,1], got {damping}")
        node_prefix = str(self.node_prefix)
        if not node_prefix:
            raise ValueError("node_prefix must be non-empty")

        object.__setattr__(self, "num_units", num_units)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "coupling_strength", coupling_strength)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "zero_diag", bool(self.zero_diag))
        object.__setattr__(self, "node_prefix", node_prefix)
        object.__setattr__(self, "initial_signal", complex(self.initial_signal))

    def store(self, *, patterns: np.ndarray, labels: list[str] | None) -> StoredMemory:
        pats = self.math.as2d(np.asarray(patterns))
        if pats.shape[1] != self.num_units:
            raise ValueError(f"patterns must have shape (K, {self.num_units})")
        pats = self.math.toPhasors(pats, dtype=self.dtype)

        count = int(pats.shape[0])
        if labels is not None and len(labels) != count:
            raise ValueError("labels length must match number of patterns")

        weights = self.weightsFromPatterns(patterns=pats)
        net = self.networkFromWeights(weights=weights)
        return StoredMemory(
            patterns=pats.astype(self.dtype, copy=False),
            weights=weights,
            labels=None if labels is None else list(labels),
            network=net,
        )

    def weightsFromPatterns(self, *, patterns: np.ndarray) -> np.ndarray:
        patterns = np.asarray(patterns)
        if patterns.ndim != 2:
            raise ValueError(f"patterns must be a 2D array, got ndim={patterns.ndim}")
        if patterns.shape[1] != self.num_units:
            raise ValueError(f"patterns must have shape (K, {self.num_units})")
        weights = (patterns.conj().T @ patterns) / float(self.num_units)
        if self.zero_diag:
            np.fill_diagonal(weights, 0.0 + 0.0j)
        return weights.astype(self.dtype, copy=False)

    def networkFromWeights(self, *, weights: np.ndarray) -> ResonantNetwork:
        weights = np.asarray(weights)
        if weights.shape != (self.num_units, self.num_units):
            raise ValueError(
                f"weights must have shape ({self.num_units}, {self.num_units}), got {weights.shape}"
            )
        net = ResonantNetwork(name="phase_associative_memory")
        net.coupling_strength = self.coupling_strength
        net.global_damping = self.damping

        # Nodes start at a constant complex signal so the substrate has a well-defined
        # initial phase/amplitude when used directly.
        for i in range(self.num_units):
            node = ResonantNode(
                node_id=self.nodeId(i),
                natural_freq=0.0,
                damping=self.damping,
                signal=self.initial_signal,
            )
            net.addNode(node)

        for src in range(self.num_units):
            for tgt in range(self.num_units):
                if self.zero_diag and src == tgt:
                    continue
                w = complex(weights[tgt, src])
                if w == 0.0 + 0.0j:
                    continue
                net.connect(source_id=self.nodeId(src), target_id=self.nodeId(tgt), weight=w, delay=0.0)
        return net

    def nodeId(self, index: int) -> str:
        return f"{self.node_prefix}{int(index)}"

