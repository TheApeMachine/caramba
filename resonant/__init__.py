"""Resonant Memory Field (RMF) substrate.

This package is the preferred name for the resonant-phase components used to
build Resonant Memory Fields (RMF).

It currently re-exports the existing reference implementation that historically
lived under the `synthnn` name.
"""

from resonant.core import ResonantNetwork, ResonantNode
from resonant.core.associative_memory import PhaseAssociativeMemory, RecallResult

__all__ = [
    "PhaseAssociativeMemory",
    "RecallResult",
    "ResonantNetwork",
    "ResonantNode",
]

