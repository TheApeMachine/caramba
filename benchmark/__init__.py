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

from .artifacts import ArtifactGenerator
from .accuracy.base import BenchmarkAccuracy
from .behavioral_v2 import BenchmarkBehavioralV2
from .latency import LatencyBenchmark
from .memory import MemoryBenchmark
from .perplexity import PerplexityBenchmark
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "PerplexityBenchmark",
    "BenchmarkAccuracy",
    "BenchmarkBehavioralV2",
    "LatencyBenchmark",
    "MemoryBenchmark",
    "ArtifactGenerator",
]
