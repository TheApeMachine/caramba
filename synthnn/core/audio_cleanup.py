"""High-level audio cleanup API for SynthNN.

This wraps the resonant-only implementation in `synthnn.core.resonant_cleanup`
with a small, stable surface that can be used from demos/CLI/manifests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthnn.core.artifacts import ArtifactProfile
from synthnn.core.cleanup_report import CleanupReport
from synthnn.core.resonant_cleanup import (
    ResonantCleanupConfig,
    cleanup_resonant_only,
    detect_artifacts_resonant,
)


@dataclass(slots=True)
class ArtifactDetector:
    """Detect tonal artifacts from audio."""

    cfg: ResonantCleanupConfig = ResonantCleanupConfig()

    def detect(self, audio: np.ndarray, *, sample_rate_hz: int) -> list[ArtifactProfile]:
        return detect_artifacts_resonant(audio, sample_rate_hz=int(sample_rate_hz), cfg=self.cfg)


@dataclass(slots=True)
class AudioCleanupEngine:
    """End-to-end audio cleanup (detect + cancel)."""

    cfg: ResonantCleanupConfig = ResonantCleanupConfig()

    def detect(self, audio: np.ndarray, *, sample_rate_hz: int) -> list[ArtifactProfile]:
        return detect_artifacts_resonant(audio, sample_rate_hz=int(sample_rate_hz), cfg=self.cfg)

    def cleanup(
        self,
        audio: np.ndarray,
        *,
        sample_rate_hz: int,
        artifacts: list[ArtifactProfile] | None = None,
    ) -> tuple[np.ndarray, CleanupReport]:
        return cleanup_resonant_only(audio, sample_rate_hz=int(sample_rate_hz), cfg=self.cfg, artifacts=artifacts)


def create_cleanup_pipeline(*, preset: str = "resonant_only") -> AudioCleanupEngine:
    """Create a cleanup engine with a reasonable preset config."""

    preset_key = str(preset).strip().lower()
    if preset_key not in {"resonant_only", "resonant"}:
        raise ValueError(f"Unknown preset: {preset!r}")
    return AudioCleanupEngine(cfg=ResonantCleanupConfig())

