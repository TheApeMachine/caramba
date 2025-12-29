from __future__ import annotations

import types
from typing import Any

import torch
from torch import nn

import benchmark.evaluators as ev


def test_latency_evaluator_builds_config_and_returns_expected_keys(monkeypatch) -> None:
    class FakeBench:
        def __init__(self, cfg: Any, device: torch.device) -> None:
            self.cfg = cfg
            self.device = device

        def run(self, model: torch.nn.Module, name: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(avg_tokens_per_second=12.5, avg_time_to_first_token_ms=3.0)

    monkeypatch.setattr(ev, "LatencyBenchmark", FakeBench)

    m = nn.Linear(2, 2)
    out = ev.LatencyEvaluator(prompt_lengths=[4], generation_lengths=[5], batch_sizes=[1]).run(
        model=m, device=torch.device("cpu"), name="m"
    )
    assert out["avg_tokens_per_second"] == 12.5
    assert out["avg_time_to_first_token_ms"] == 3.0


def test_memory_evaluator_handles_kvcache_analysis_presence(monkeypatch) -> None:
    class FakeAnalysis:
        bytes_per_token_fp16: float = 2.0
        bytes_per_token_dba_fp16: float = 1.0

    class FakeBench:
        def __init__(self, cfg: Any, device: torch.device) -> None:
            self.cfg = cfg
            self.device = device

        def run(self, model: torch.nn.Module, name: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                measurements=[types.SimpleNamespace(model_memory_mb=7.0)],
                peak_memory_mb=10.0,
                kvcache_analysis=FakeAnalysis(),
            )

    monkeypatch.setattr(ev, "MemoryBenchmark", FakeBench)

    m = nn.Linear(2, 2)
    out = ev.MemoryEvaluator(sequence_lengths=[4], batch_sizes=[1]).run(
        model=m, device=torch.device("cpu"), name="m"
    )
    assert out["model_memory_mb"] == 7.0
    assert out["peak_memory_mb"] == 10.0
    assert out["kvcache_bytes_per_token_fp16"] == 2.0
    assert out["kvcache_bytes_per_token_dba_fp16"] == 1.0

