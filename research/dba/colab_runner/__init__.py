"""Colab runner

This module allows you to run workloads on Google Colab.
"""
from __future__ import annotations

from research.dba.colab_runner.base import ColabRunnerBase
from research.dba.colab_runner.benchmark import BenchmarkConfig, BenchmarkRunner
from research.dba.colab_runner.browser import PlaywrightRunner
from research.dba.colab_runner.cells import get_benchmark_cells, load_cell
from research.dba.colab_runner.colab import ColabRunner
from research.dba.colab_runner.javascript import JavaScriptRunner
from research.dba.colab_runner.notebook import NotebookRunner
from research.dba.colab_runner.paths import normalize_drive_path

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "ColabRunnerBase",
    "ColabRunner",
    "JavaScriptRunner",
    "NotebookRunner",
    "PlaywrightRunner",
    "get_benchmark_cells",
    "load_cell",
    "normalize_drive_path",
]
