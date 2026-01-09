"""Artifact schema for audio cleanup.

This module intentionally stays lightweight (NumPy only) so it can be used in
Caramba's core runtime without pulling in heavy audio stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArtifactType(str, Enum):
    """High-level artifact categories."""

    HUM = "hum"
    BUZZ = "buzz"
    WHINE = "whine"
    TONAL_NOISE = "tonal_noise"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ArtifactProfile:
    """A detected tonal artifact in an audio stream."""

    artifact_type: ArtifactType
    frequency_hz: float
    bandwidth_hz: float
    strength: float
    phase_coherence: float
    persistence: float
    harmonics_hz: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": str(self.artifact_type.value),
            "frequency_hz": float(self.frequency_hz),
            "bandwidth_hz": float(self.bandwidth_hz),
            "strength": float(self.strength),
            "phase_coherence": float(self.phase_coherence),
            "persistence": float(self.persistence),
            "harmonics_hz": [float(h) for h in self.harmonics_hz],
            "metadata": dict(self.metadata),
        }

