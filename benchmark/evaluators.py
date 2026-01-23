"""Engine-level evaluators.

These evaluate runtime properties (latency/memory). They are not task-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from benchmark.latency import LatencyBenchmark
from benchmark.memory import MemoryBenchmark
from config.benchmark import LatencyBenchmarkConfig, MemoryBenchmarkConfig


@dataclass(frozen=True, slots=True)
class LatencyEvaluator:
    prompt_lengths: list[int] | None = None
    generation_lengths: list[int] | None = None
    batch_sizes: list[int] | None = None
    warmup_runs: int = 3
    timed_runs: int = 10
    use_cache: bool = True
    cache_kind: str = "fp16"
    seed: int = 42

    def run(
        self, *, model: torch.nn.Module, device: torch.device, name: str = "model"
    ) -> dict[str, Any]:
        cfg = LatencyBenchmarkConfig(
            seed=int(self.seed),
            prompt_lengths=list(self.prompt_lengths or [128, 512, 1024, 2048]),
            generation_lengths=list(self.generation_lengths or [128, 256, 512]),
            batch_sizes=list(self.batch_sizes or [1, 4, 8]),
            warmup_runs=int(self.warmup_runs),
            timed_runs=int(self.timed_runs),
            use_cache=bool(self.use_cache),
            cache_kind=str(self.cache_kind),
        )
        bench = LatencyBenchmark(cfg, device)
        r = bench.run(model, name)
        return {
            "avg_tokens_per_second": float(r.avg_tokens_per_second),
            "avg_time_to_first_token_ms": float(r.avg_time_to_first_token_ms),
        }


@dataclass(frozen=True, slots=True)
class MemoryEvaluator:
    sequence_lengths: list[int] | None = None
    batch_sizes: list[int] | None = None
    measure_peak: bool = True
    measure_kvcache: bool = True
    quantization_modes: list[str] | None = None
    seed: int = 42

    def run(
        self, *, model: torch.nn.Module, device: torch.device, name: str = "model"
    ) -> dict[str, Any]:
        cfg = MemoryBenchmarkConfig(
            seed=int(self.seed),
            sequence_lengths=list(self.sequence_lengths or [512, 1024, 2048, 4096]),
            batch_sizes=list(self.batch_sizes or [1, 4, 8]),
            measure_peak=bool(self.measure_peak),
            measure_kvcache=bool(self.measure_kvcache),
            quantization_modes=list(self.quantization_modes or ["fp16", "q8", "q4"]),
        )
        bench = MemoryBenchmark(cfg, device)
        r = bench.run(model, name)
        model_mem = max((m.model_memory_mb for m in r.measurements), default=0.0)
        out: dict[str, Any] = {
            "model_memory_mb": float(model_mem),
            "peak_memory_mb": float(r.peak_memory_mb),
        }
        if r.kvcache_analysis:
            out["kvcache_bytes_per_token_fp16"] = float(r.kvcache_analysis.bytes_per_token_fp16)
            if r.kvcache_analysis.bytes_per_token_dba_fp16 is not None:
                out["kvcache_bytes_per_token_dba_fp16"] = float(r.kvcache_analysis.bytes_per_token_dba_fp16)
        return out

