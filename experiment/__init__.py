"""Unified experiment runner.

Orchestrates the complete experiment pipeline:
1. Model loading/creation from manifest
2. Upcycle (attention surgery + blockwise distillation)
3. Benchmarking (teacher vs student comparison)
4. Artifact generation (CSV, charts, LaTeX tables)

This is the main entry point for running experiments end-to-end.
"""
from __future__ import annotations

from experiment.runner import ExperimentRunner
from experiment.group import ExperimentGroup
from experiment.benchmarks import ExperimentBenchmarks
from experiment.results import ExperimentResults

__all__ = [
    "ExperimentRunner",
    "ExperimentGroup",
    "ExperimentBenchmarks",
    "ExperimentResults",
]
