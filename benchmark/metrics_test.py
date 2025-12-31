from __future__ import annotations

import types
from typing import Any

import torch
from torch import nn

import caramba.benchmark.metrics as metrics


def test_perplexity_metric_wraps_benchmark_result(monkeypatch) -> None:
    class FakeBench:
        def __init__(self, cfg: Any, device: Any) -> None:
            self.cfg = cfg
            self.device = device

        def run(self, model: Any, name: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(perplexity=123.0, loss=4.5)

    monkeypatch.setattr(metrics, "PerplexityBenchmark", FakeBench)

    m = nn.Linear(2, 2)
    out = metrics.PerplexityMetric(dataset="x", block_size=8, batch_size=2).run(
        model=m, device=torch.device("cpu"), name="m"
    )
    assert out == {"perplexity": 123.0, "loss": 4.5}

