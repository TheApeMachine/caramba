"""Memory benchmark: measuring KV-cache and peak memory usage.

For DBA upcycling, the key metric is KV-cache memory reduction. Standard
attention caches K and V tensors of size n_kv_heads × head_dim per token.
DBA caches semantic keys (sem_dim), geometric keys (geo_dim), and values
(v_dim)—typically much smaller total.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import cast


import torch
from torch import nn

from benchmark.utils import get_model_vocab_size
from config.benchmark import MemoryBenchmarkConfig
from caramba.layer.attention import AttentionLayer, AttentionMode

logger = logging.getLogger(__name__)

def _rss_mb() -> float | None:
    """Best-effort process RSS/peak RSS in MB (platform-dependent)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        x = float(getattr(r, "ru_maxrss"))
        if x <= 0:
            return None
        # macOS: bytes. Linux: kilobytes.
        if sys.platform == "darwin":
            return x / (1024.0 * 1024.0)
        return x / 1024.0
    except Exception:
        return None


@dataclass
class MemoryMeasurement:
    """Single memory measurement for a specific configuration."""

    # Deterministic input provenance (audit trail).
    seed: int
    input_ids: list[list[int]]
    input_ids_sha256: str

    seq_len: int
    batch_size: int
    peak_memory_mb: float
    kvcache_memory_mb: float
    model_memory_mb: float
    quantization: str

    # Optional backend telemetry (does not affect scoring; used for audits).
    mps_driver_peak_memory_mb: float | None = None
    mps_recommended_max_mb: float | None = None


@dataclass
class KVCacheAnalysis:
    """Analysis of KV-cache memory usage.

    For standard attention, K and V both use n_kv_heads × head_dim.
    For DBA, the cache stores:
    - k_sem: semantic keys (sem_dim per layer)
    - k_geo: geometric keys (geo_dim per layer)
    - v: values (v_dim per layer)

    Byte estimates include quantization overhead:
    - fp16: 2 bytes/element
    - q8: 1 byte/element
    - q4: 0.625 bytes/element
    """

    model_name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    attention_mode: str
    bytes_per_token_fp16: float
    bytes_per_token_q8: float
    bytes_per_token_q4: float

    # DBA-specific dimensions
    sem_dim: int | None = None
    geo_dim: int | None = None
    v_dim: int | None = None
    bytes_per_token_dba_fp16: float | None = None
    bytes_per_token_dba_q8: float | None = None
    bytes_per_token_dba_q4: float | None = None


@dataclass
class MemoryResult:
    """Complete memory benchmark results for a model."""

    model_name: str
    measurements: list[MemoryMeasurement] = field(default_factory=list)
    kvcache_analysis: KVCacheAnalysis | None = None

    @property
    def peak_memory_mb(self) -> float:
        """Maximum peak memory across all measurements."""
        if not self.measurements:
            return 0.0
        return max(m.peak_memory_mb for m in self.measurements)

    @property
    def peak_allocated_mb(self) -> float:
        """Backwards-compatible alias for peak_memory_mb."""
        return self.peak_memory_mb


class MemoryBenchmark:
    """Measures memory usage including KV-cache analysis.

    Analyzes the model architecture to compute theoretical KV-cache sizes,
    then runs actual forward passes to measure peak memory.
    """

    def __init__(
        self, config: MemoryBenchmarkConfig, device: torch.device
    ) -> None:
        """Set up the benchmark with config and target device."""
        self.config = config
        self.device = device

    def run(
        self,
        model: nn.Module,
        model_name: str,
        *,
        on_measurement=None,
    ) -> MemoryResult:
        """Run the memory benchmark, measuring both theoretical and actual usage."""
        model.eval()
        result = MemoryResult(model_name=model_name)

        # Analyze KV-cache structure from model architecture
        result.kvcache_analysis = self._analyze_kvcache(model, model_name)

        # Measure actual memory usage
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for quant in self.config.quantization_modes:
                    measurement = self._measure(
                        model=model,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        quantization=quant,
                    )
                    result.measurements.append(measurement)
                    if on_measurement is not None:
                        try:
                            on_measurement(measurement)
                        except Exception:
                            pass

        return result

    def _analyze_kvcache(self, model: nn.Module, model_name: str) -> KVCacheAnalysis:
        """Analyze KV-cache structure from model architecture.

        Inspects attention layers to determine cache dimensions and
        compute theoretical bytes per token for different precisions.
        """
        n_layers = 0
        n_kv_heads: int | None = None
        head_dim: int | None = None
        attention_mode: str | None = None
        sem_dim: int | None = None
        geo_dim: int | None = None
        v_dim: int | None = None

        for module in model.modules():
            if isinstance(module, AttentionLayer):
                # Type assertion to help type checker narrow the type
                attn_layer = cast(AttentionLayer, module)
                n_layers += 1
                # Explicit type conversions to satisfy type checker
                layer_n_kv_heads = int(attn_layer.n_kv_heads)
                layer_head_dim = int(attn_layer.head_dim)
                layer_mode = str(attn_layer.mode.value)

                # Validate consistency across layers
                if n_kv_heads is None:
                    n_kv_heads = layer_n_kv_heads
                elif n_kv_heads != layer_n_kv_heads:
                    raise ValueError(
                        f"Inconsistent n_kv_heads: {n_kv_heads} vs {layer_n_kv_heads}"
                    )

                if head_dim is None:
                    head_dim = layer_head_dim
                elif head_dim != layer_head_dim:
                    raise ValueError(
                        f"Inconsistent head_dim: {head_dim} vs {layer_head_dim}"
                    )

                if attention_mode is None:
                    attention_mode = layer_mode
                elif attention_mode != layer_mode:
                    raise ValueError(
                        f"Inconsistent attention mode: {attention_mode} vs {layer_mode}"
                    )

                # Extract DBA dimensions
                if attn_layer.mode == AttentionMode.DECOUPLED:
                    cfg = attn_layer.config
                    # Convert to int | None explicitly
                    layer_sem_dim = int(cfg.sem_dim) if cfg.sem_dim is not None else None
                    layer_geo_dim = int(cfg.geo_dim) if cfg.geo_dim is not None else None
                    layer_v_dim = int(cfg.v_dim) if cfg.v_dim is not None else None

                    # Adjust for GQA: cache stores n_kv_heads * head_dim, but config has n_heads * head_dim
                    n_heads = int(cfg.n_heads)
                    kv_heads = int(cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads)
                    
                    if layer_sem_dim is not None:
                        layer_sem_dim = (layer_sem_dim // n_heads) * kv_heads
                    if layer_geo_dim is not None:
                        layer_geo_dim = (layer_geo_dim // n_heads) * kv_heads
                    if layer_v_dim is not None:
                        layer_v_dim = (layer_v_dim // n_heads) * kv_heads

                    if sem_dim is None:
                        sem_dim = layer_sem_dim
                    elif sem_dim != layer_sem_dim:
                        raise ValueError(
                            f"Inconsistent sem_dim: {sem_dim} vs {layer_sem_dim}"
                        )

                    if geo_dim is None:
                        geo_dim = layer_geo_dim
                    elif geo_dim != layer_geo_dim:
                        raise ValueError(
                            f"Inconsistent geo_dim: {geo_dim} vs {layer_geo_dim}"
                        )

                    if v_dim is None:
                        v_dim = layer_v_dim
                    elif v_dim != layer_v_dim:
                        raise ValueError(
                            f"Inconsistent v_dim: {v_dim} vs {layer_v_dim}"
                        )

        # Use defaults if no attention layers found
        used_defaults = []
        if n_kv_heads is None:
            n_kv_heads = 0
            used_defaults.append("n_kv_heads=0")
        if head_dim is None:
            head_dim = 0
            used_defaults.append("head_dim=0")
        if attention_mode is None:
            attention_mode = "standard"
            used_defaults.append("attention_mode='standard'")

        if n_layers == 0 or used_defaults:
            logger.warning(
                "No attention layers detected; using defaults: %s",
                ", ".join(used_defaults) if used_defaults else "n_layers=0",
            )

        # Calculate bytes per token for standard attention
        kv_dim = n_kv_heads * head_dim
        bytes_fp16 = 2 * n_layers * kv_dim * 2.0
        bytes_q8 = 2 * n_layers * kv_dim * 1.0
        bytes_q4 = 2 * n_layers * kv_dim * 0.625

        # Calculate DBA bytes per token
        bytes_dba_fp16: float | None = None
        bytes_dba_q8: float | None = None
        bytes_dba_q4: float | None = None
        if sem_dim is not None and geo_dim is not None:
            actual_v_dim = v_dim if v_dim is not None else kv_dim
            dba_elements_per_token = n_layers * (sem_dim + geo_dim + actual_v_dim)
            bytes_dba_fp16 = float(dba_elements_per_token * 2.0)
            bytes_dba_q8 = float(dba_elements_per_token * 1.0)
            bytes_dba_q4 = float(dba_elements_per_token * 0.625)

        return KVCacheAnalysis(
            model_name=model_name,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            attention_mode=attention_mode,
            bytes_per_token_fp16=float(bytes_fp16),
            bytes_per_token_q8=float(bytes_q8),
            bytes_per_token_q4=float(bytes_q4),
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            v_dim=v_dim,
            bytes_per_token_dba_fp16=bytes_dba_fp16,
            bytes_per_token_dba_q8=bytes_dba_q8,
            bytes_per_token_dba_q4=bytes_dba_q4,
        )

    def _measure(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        quantization: str,
    ) -> MemoryMeasurement:
        """Measure actual memory usage for a specific configuration."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model_memory = float(torch.cuda.memory_allocated() / (1024 * 1024))
            mps_driver_peak: float | None = None
            mps_recommended: float | None = None
        elif self.device.type == "mps":
            # MPS does not expose reset_peak_memory_stats(). We instead sample
            # allocated + driver allocated during the measurement window.
            torch.mps.empty_cache()
            try:
                model_memory = float(torch.mps.current_allocated_memory() / (1024 * 1024))
            except Exception as e:
                raise RuntimeError(f"MemoryBenchmark: failed to read MPS current_allocated_memory: {e!r}") from e
            try:
                mps_recommended = float(torch.mps.recommended_max_memory() / (1024 * 1024))
            except Exception:
                mps_recommended = None
            # Initialize peak trackers with baseline.
            try:
                mps_driver_peak = float(torch.mps.driver_allocated_memory() / (1024 * 1024))
            except Exception:
                mps_driver_peak = None
        else:
            # CPU fallback is explicitly RSS-based (not GPU peak alloc).
            rss0 = _rss_mb()
            if rss0 is None:
                raise RuntimeError("MemoryBenchmark: failed to read RSS for CPU memory measurement.")
            model_memory = float(rss0)
            mps_driver_peak = None
            mps_recommended = None

        vocab_size = self._resolve_effective_vocab_size(model)

        seed = int(self.config.seed) * 1000003 + int(batch_size) * 1009 + int(seq_len) * 9176
        seed = int(seed % (2**31 - 1))
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        input_ids = torch.randint(
            0,
            int(vocab_size),
            (int(batch_size), int(seq_len)),
            generator=g,
            device=self.device,
            dtype=torch.long,
        )
        input_ids_list = input_ids.detach().cpu().tolist()
        h = hashlib.sha256()
        for row in input_ids_list:
            h.update((",".join(str(int(x)) for x in row) + "\n").encode("utf-8"))
        input_ids_sha = h.hexdigest()

        with torch.no_grad():
            _ = model(input_ids)

        if self.device.type == "cuda":
            peak_memory = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        elif self.device.type == "mps":
            # Sample MPS allocated/driver allocated after the forward.
            try:
                peak_memory = float(torch.mps.current_allocated_memory() / (1024 * 1024))
            except Exception as e:
                raise RuntimeError(f"MemoryBenchmark: failed to read MPS current_allocated_memory: {e!r}") from e
            try:
                drv = float(torch.mps.driver_allocated_memory() / (1024 * 1024))
                if mps_driver_peak is None:
                    mps_driver_peak = drv
                else:
                    mps_driver_peak = max(float(mps_driver_peak), drv)
            except Exception:
                # Keep whatever we already have; driver stats may not be available on all builds.
                pass
        else:
            rss1 = _rss_mb()
            if rss1 is None:
                raise RuntimeError("MemoryBenchmark: failed to read RSS for CPU memory measurement.")
            peak_memory = float(rss1)

        kvcache_memory = self._estimate_kvcache_memory(
            model=model,
            batch_size=batch_size,
            seq_len=seq_len,
            quantization=quantization,
        )

        # Sanity checks (zero impact on measured values; fail-fast on broken runs).
        if not math.isfinite(float(peak_memory)) or float(peak_memory) <= 0.0:
            raise ValueError(
                f"MemoryBenchmark: peak_memory_mb must be finite and > 0, got {peak_memory!r} "
                f"(device={self.device.type}, batch_size={batch_size}, seq_len={seq_len})."
            )
        if not math.isfinite(float(kvcache_memory)) or float(kvcache_memory) <= 0.0:
            raise ValueError(
                f"MemoryBenchmark: kvcache_memory_mb must be finite and > 0, got {kvcache_memory!r} "
                f"(batch_size={batch_size}, seq_len={seq_len}, quant={quantization})."
            )

        return MemoryMeasurement(
            seed=int(seed),
            input_ids=input_ids_list,
            input_ids_sha256=str(input_ids_sha),
            seq_len=seq_len,
            batch_size=batch_size,
            peak_memory_mb=peak_memory,
            kvcache_memory_mb=kvcache_memory,
            model_memory_mb=model_memory,
            quantization=quantization,
            mps_driver_peak_memory_mb=float(mps_driver_peak) if mps_driver_peak is not None else None,
            mps_recommended_max_mb=float(mps_recommended) if mps_recommended is not None else None,
        )

    def _estimate_kvcache_memory(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        quantization: str,
    ) -> float:
        """Estimate theoretical KV-cache memory usage."""
        n_layers = 0
        kv_dim = 0
        dba_k_dim: int | None = None
        dba_v_dim: int | None = None

        for module in model.modules():
            if isinstance(module, AttentionLayer):
                # Type assertion to help type checker narrow the type
                attn_layer = cast(AttentionLayer, module)
                n_layers += 1
                # Explicit type conversions to satisfy type checker
                kv_dim = int(attn_layer.n_kv_heads) * int(attn_layer.head_dim)

                if attn_layer.mode == AttentionMode.DECOUPLED:
                    cfg = attn_layer.config
                    if cfg.sem_dim is not None and cfg.geo_dim is not None:
                        # Convert to int explicitly
                        n_heads = int(cfg.n_heads)
                        kv_heads = int(cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads)
                        
                        sem_total = int(cfg.sem_dim)
                        geo_total = int(cfg.geo_dim)
                        
                        dba_k_dim = ((sem_total // n_heads) * kv_heads) + ((geo_total // n_heads) * kv_heads)
                        
                        v_total = int(cfg.v_dim) if cfg.v_dim is not None else kv_dim # Fallback might be wrong if v_dim not set?
                        if cfg.v_dim is not None:
                             dba_v_dim = (int(cfg.v_dim) // n_heads) * kv_heads
                        else:
                             dba_v_dim = None # Will fallback to kv_dim logic below which is correct for standard V
                    else:
                        dba_k_dim = None
                        dba_v_dim = None

        bytes_per_elem = {
            "fp16": 2.0,
            "fp32": 4.0,
            "q8": 1.0,
            "q8_0": 1.0,
            "q4": 0.625,
            "q4_0": 0.625,
            "nf4": 0.625,
        }.get(quantization, 2.0)

        if dba_k_dim is not None:
            actual_v_dim = dba_v_dim if dba_v_dim is not None else kv_dim
            k_bytes = float(n_layers * batch_size * seq_len * dba_k_dim * bytes_per_elem)
            v_bytes = float(n_layers * batch_size * seq_len * actual_v_dim * bytes_per_elem)
            total_bytes = k_bytes + v_bytes
        else:
            total_bytes = float(
                2 * n_layers * batch_size * seq_len * kv_dim * bytes_per_elem
            )

        return float(total_bytes / (1024 * 1024))

    def _get_vocab_size(self, model: nn.Module) -> int:
        """Get vocab size from model."""
        return get_model_vocab_size(model, default=32000)

    def _resolve_effective_vocab_size(self, model: nn.Module) -> int:
        model_vocab = int(self._get_vocab_size(model))
        vv = getattr(self.config, "valid_vocab_size", None)
        if vv is None:
            return model_vocab
        eff = int(vv)
        if eff > model_vocab:
            raise ValueError(
                "MemoryBenchmark: valid_vocab_size exceeds model vocab_size "
                f"(valid_vocab_size={eff}, model_vocab_size={model_vocab})."
            )
        return eff
