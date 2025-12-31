"""Metric components.

These are manifest-referenced metrics that can be run after/beside training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from caramba.benchmark.perplexity import PerplexityBenchmark
from caramba.config.benchmark import PerplexityBenchmarkConfig


@dataclass(frozen=True, slots=True)
class PerplexityMetric:
    """Compute perplexity on a token dataset for a language model module."""

    dataset: str
    block_size: int = 2048
    batch_size: int = 1
    num_batches: int | None = None
    stride: int | None = None

    def run(
        self, *, model: torch.nn.Module, device: torch.device, name: str = "model"
    ) -> dict[str, float]:
        cfg = PerplexityBenchmarkConfig(
            dataset=self.dataset,
            block_size=int(self.block_size),
            batch_size=int(self.batch_size),
            num_batches=self.num_batches,
            stride=self.stride,
        )
        bench = PerplexityBenchmark(cfg, device)
        r = bench.run(model, name)
        return {"perplexity": float(r.perplexity), "loss": float(r.loss)}

