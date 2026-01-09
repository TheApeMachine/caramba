"""Context benchmark result

Aggregates all measurements from a context benchmark run, storing both
sweep measurements (one per context length) and decode measurements
(throughput at each context length). Provides a complete picture of
model performance across different context lengths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from collector.measurement.context.sweep import ContextSweepMeasurement
from collector.measurement.context.decode import ContextDecodeMeasurement


@dataclass
class ContextResult:
    """Context benchmark result

    Aggregates all measurements from a context benchmark run, storing both
    sweep measurements (one per context length) and decode measurements
    (throughput at each context length). Provides a complete picture of
    model performance across different context lengths.
    """
    model_name: str
    sweep: list[ContextSweepMeasurement] = field(default_factory=list)
    decode: list[ContextDecodeMeasurement] = field(default_factory=list)