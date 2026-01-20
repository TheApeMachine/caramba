"""Accuracy benchmark package

Includes:
- BenchmarkAccuracy: Main accuracy benchmark runner
- AccuracyResult: Result container
- Artifact generation: JSON, CSV, LaTeX, markdown, visualizations
"""
from __future__ import annotations

from .base import BenchmarkAccuracy
from collector.measurement.accuracy.result import AccuracyResult
from .utils import DictCoercion, TextNormalization
from eval.logprob.scorer import LogprobScorer
from .artifacts import (
    AccuracyArtifactConfig,
    AccuracyArtifactGenerator,
    MultiModelAccuracyResults,
    generate_accuracy_artifacts,
)

__all__ = [
    "BenchmarkAccuracy",
    "AccuracyResult",
    "LogprobScorer",
    "DictCoercion",
    "TextNormalization",
    # Artifact generation
    "AccuracyArtifactConfig",
    "AccuracyArtifactGenerator",
    "MultiModelAccuracyResults",
    "generate_accuracy_artifacts",
]
