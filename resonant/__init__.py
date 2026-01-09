"""Resonant Memory Field (RMF) substrate.

This package is the preferred name for the resonant-phase components used to
build Resonant Memory Fields (RMF).

This package provides the reference resonant-phase implementation (formerly
named `synthnn`).
"""

from resonant.core import ResonantNetwork, ResonantNode
from resonant.core.associative_memory import PhaseAssociativeMemory, RecallResult
from resonant.core.artifacts import ArtifactProfile, ArtifactType
from resonant.core.audio_cleanup import ArtifactDetector, AudioCleanupEngine, create_cleanup_pipeline
from resonant.core.cleanup_report import CleanupReport
from resonant.core.resonant_cleanup import ResonantCleanupConfig, cleanup_resonant_only, detect_artifacts_resonant

__all__ = [
    "ArtifactDetector",
    "ArtifactProfile",
    "ArtifactType",
    "AudioCleanupEngine",
    "CleanupReport",
    "PhaseAssociativeMemory",
    "RecallResult",
    "ResonantCleanupConfig",
    "ResonantNetwork",
    "ResonantNode",
    "cleanup_resonant_only",
    "create_cleanup_pipeline",
    "detect_artifacts_resonant",
]

