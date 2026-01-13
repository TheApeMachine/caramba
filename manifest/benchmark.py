"""Benchmark module

The benchmark module contains the benchmark for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, PositiveInt


class BenchmarkType(str, Enum):
    """Benchmark type enumeration."""
    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    GENERATION = "generation"
    BEHAVIOR = "behavior"
    CONTEXT = "context"
    WINOGRANDE = "winogrande"
    HALLSWAG = "hellaswag"
    PIQA = "piqa"
    MMLU = "mmlu"
    BIG_BENCH = "big_bench"
    SFT_DATA = "sft_data"
    PPO_DATA = "ppo_data"


class Benchmark(BaseModel):
    """Benchmark configuration."""
    type: BenchmarkType = Field(..., description="Benchmark type")