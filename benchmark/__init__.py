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

from caramba.benchmark.artifacts import ArtifactGenerator
from caramba.benchmark.accuracy.base import BenchmarkAccuracy
from caramba.benchmark.behavioral_v2 import BenchmarkBehavioralV2
from caramba.benchmark.latency import LatencyBenchmark
from caramba.benchmark.memory import MemoryBenchmark
from caramba.benchmark.perplexity import PerplexityBenchmark
from caramba.benchmark.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "PerplexityBenchmark",
    "BenchmarkAccuracy",
    "BenchmarkBehavioralV2",
    "LatencyBenchmark",
    "MemoryBenchmark",
    "ArtifactGenerator",
]
