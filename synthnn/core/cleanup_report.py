"""Cleanup report schema.

The report is designed to be serializable (e.g. JSON) and easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synthnn.core.artifacts import ArtifactProfile


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Summary of an audio cleanup pass."""

    sample_rate_hz: int
    num_samples: int
    artifacts: list[ArtifactProfile] = field(default_factory=list)
    rms_before: float = 0.0
    rms_after: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        sr = max(1, int(self.sample_rate_hz))
        return float(int(self.num_samples) / sr)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": int(self.sample_rate_hz),
            "num_samples": int(self.num_samples),
            "duration_s": float(self.duration_s),
            "rms_before": float(self.rms_before),
            "rms_after": float(self.rms_after),
            "artifacts": [a.as_dict() for a in self.artifacts],
            "notes": dict(self.notes),
        }

