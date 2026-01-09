"""Context decode measurement

Captures throughput metrics for decoding at a specific context length,
including warmup iterations and timed decode throughput. Measures how well
the model maintains performance when generating tokens after a long context.
"""
from __future__ import annotations

from dataclasses import dataclass

from collector.measurement.context.base import ContextMeasurement


@dataclass
class ContextDecodeMeasurement(ContextMeasurement):
    """Context decode measurement result

    Captures throughput metrics for decoding at a specific context length,
    including warmup iterations and timed decode throughput. Measures how well
    the model maintains performance when generating tokens after a long context.
    """
    context_len: int
    chunk_size_used: int
    batch_size: int
    decode_len: int
    decode_warmup: int
    prefill_total_s: float
    decode_total_ms: float
    decode_tok_per_s: float
    ok: bool