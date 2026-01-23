"""Benchmarking utilities for model comparison and paper-ready artifacts.

After upcycling, we need to measure what matters: Did we preserve quality?
Did we reduce memory? How fast is generation? This package provides
benchmarks that answer these questions and generate paper-ready outputs.

Benchmark types:
- Perplexity: Language modeling quality
- Latency: Tokens per second, time to first token
- Memory: KV-cache and peak memory usage
- Artifacts: CSV, JSON, PNG, and LaTeX outputs
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import for static type checking only. Runtime imports are lazy to avoid
    # importing heavy dependencies (e.g. torch) at package import time.
    from .artifacts import ArtifactGenerator
    from .accuracy.base import BenchmarkAccuracy
    from .latency import LatencyBenchmark
    from .memory import MemoryBenchmark
    from .perplexity import PerplexityBenchmark
    from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "PerplexityBenchmark",
    "BenchmarkAccuracy",
    "LatencyBenchmark",
    "MemoryBenchmark",
    "ArtifactGenerator",
]


def __getattr__(name: str) -> Any:
    if name == "BenchmarkRunner":
        from .runner import BenchmarkRunner as _BenchmarkRunner

        return _BenchmarkRunner
    if name == "PerplexityBenchmark":
        from .perplexity import PerplexityBenchmark as _PerplexityBenchmark

        return _PerplexityBenchmark
    if name == "BenchmarkAccuracy":
        from .accuracy.base import BenchmarkAccuracy as _BenchmarkAccuracy

        return _BenchmarkAccuracy
    if name == "LatencyBenchmark":
        from .latency import LatencyBenchmark as _LatencyBenchmark

        return _LatencyBenchmark
    if name == "MemoryBenchmark":
        from .memory import MemoryBenchmark as _MemoryBenchmark

        return _MemoryBenchmark
    if name == "ArtifactGenerator":
        from .artifacts import ArtifactGenerator as _ArtifactGenerator

        return _ArtifactGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
