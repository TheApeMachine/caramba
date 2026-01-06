"""Associative memory state

Holds the stored attractors and derived structures used during recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthnn.core.resonant_network import ResonantNetwork


@dataclass(slots=True, eq=False)
class StoredMemory:
    """Stored memory payload

    Separates learned/derived state from the public PhaseAssociativeMemory API.
    """

    patterns: np.ndarray
    weights: np.ndarray
    labels: list[str] | None
    network: ResonantNetwork

