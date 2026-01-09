"""Context sweep measurement

Captures performance metrics for a single context length sweep, including
prefill timing, single-token decode timing, and loss/perplexity on the
final chunk. Used to track how model performance scales with context length.
"""
from __future__ import annotations

from dataclasses import dataclass

from collector.measurement.context.base import ContextMeasurement


@dataclass
class ContextSweepMeasurement(ContextMeasurement):
    """Context sweep measurement result

    Captures performance metrics for a single context length sweep, including
    prefill timing, single-token decode timing, and loss/perplexity on the
    final chunk. Used to track how model performance scales with context length.
    """
    context_len: int
    chunk_size_used: int
    batch_size: int
    prefill_total_s: float
    prefill_last_chunk_ms: float
    decode_one_ms: float
    decode_one_tok_per_s: float
    loss_last_chunk: float
    ppl_last_chunk: float
    ok: bool