"""SynthNN core

Exports the stable public surface for SynthNN building blocks.
"""

from synthnn.core.artifacts import ArtifactProfile, ArtifactType
from synthnn.core.associative_memory.memory import PhaseAssociativeMemory
from synthnn.core.audio_cleanup import ArtifactDetector, AudioCleanupEngine, create_cleanup_pipeline
from synthnn.core.cleanup_report import CleanupReport
from synthnn.core.resonant_cleanup import ResonantCleanupConfig, cleanup_resonant_only, detect_artifacts_resonant
from synthnn.core.resonant_network import ResonantNetwork
from synthnn.core.resonant_node import ResonantNode

__all__ = [
    "ArtifactDetector",
    "ArtifactProfile",
    "ArtifactType",
    "AudioCleanupEngine",
    "CleanupReport",
    "PhaseAssociativeMemory",
    "ResonantCleanupConfig",
    "ResonantNetwork",
    "ResonantNode",
    "cleanup_resonant_only",
    "create_cleanup_pipeline",
    "detect_artifacts_resonant",
]

