"""Context sweep measurement

Captures performance metrics for a single context length sweep, including
prefill timing, single-token decode timing, and loss/perplexity metrics.
Used to track how model performance scales with context length.
"""
from __future__ import annotations

from dataclasses import dataclass

from collector.measurement.context.base import ContextMeasurement


@dataclass
class ContextSweepMeasurement(ContextMeasurement):
    """Context sweep measurement result

    Captures performance metrics for a single context length sweep, including
    prefill timing, single-token decode timing, and loss/perplexity metrics.
    Used to track how model performance scales with context length.

    Loss/PPL fields:
    - loss/ppl: Accumulated across ALL chunks (standard perplexity metric)
    - loss_last_chunk/ppl_last_chunk: Only the final chunk (legacy, for debugging)
    """
    context_len: int
    chunk_size_used: int
    batch_size: int
    prefill_total_s: float
    prefill_last_chunk_ms: float
    decode_one_ms: float
    decode_one_tok_per_s: float
    loss: float  # Accumulated across all chunks
    ppl: float   # Accumulated across all chunks
    loss_last_chunk: float  # Legacy: final chunk only
    ppl_last_chunk: float   # Legacy: final chunk only
    ok: bool

    # ------------------------------------------------------------------
    # Optional telemetry (best-effort; may be None on some platforms)
    # ------------------------------------------------------------------
    rss_mb_before: float | None = None
    rss_mb_after: float | None = None
    mps_allocated_mb_before: float | None = None
    mps_allocated_mb_after: float | None = None
    mps_driver_allocated_mb_before: float | None = None
    mps_driver_allocated_mb_after: float | None = None
    mps_recommended_max_mb: float | None = None