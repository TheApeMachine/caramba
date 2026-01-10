"""Telemetry structures for memory health and dynamics.

These structures provide the "sensors" for the UniversalMemoryTuner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ResonantSettlingMetrics:
    """Metrics from one resonant routing forward pass."""
    final_sim: float           # Max similarity across all buckets
    convergence_steps: int      # Number of steps until similarity stabilized
    energy_drop: float          # Change in system energy (Lyapunov proxy)
    bucket_entropy: float       # Diversity of bucket assignments (B, T, H)
    state_drift: float          # Deviation from unit-circle normalization


@dataclass(frozen=True, slots=True)
class VsaNoveltyMetrics:
    """Metrics from VSA tag matching and novelty filtering."""
    novelty_ema: float          # Moving average of "newness"
    write_rejection_rate: float # Fraction of writes blocked by novelty threshold
    match_confidence: float     # Gap between best and second-best slot match
    tag_collision_rate: float   # Fraction of slots with similar tags but diff values


@dataclass(frozen=True, slots=True)
class RmfDynamicsMetrics:
    """Metrics from Resonant Memory Field (Successor Bias)."""
    field_rms: float            # Root-mean-square magnitude of the RMF field
    delta_rms: float            # RMS of the bias injected into routing
    prediction_error: float     # Error between field-prediction and actual lookup
    bias_ratio: float           # ||delta|| / ||raw_tag||


@dataclass
class MemoryHealthTelemetry:
    """Holistic view of memory health at a specific step."""
    step: int = 0
    utilization: float = 0.0    # % of buckets with at least one non-empty slot
    conflict_rate: float = 0.0  # % of writes that update an existing non-empty slot
    
    # Nested component metrics
    resonant: ResonantSettlingMetrics | None = None
    vsa: VsaNoveltyMetrics | None = None
    rmf: RmfDynamicsMetrics | None = None

    # Custom auxiliary metrics
    aux: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten for logging to WandB/TelemetryStream."""
        out = {
            "memory/utilization": self.utilization,
            "memory/conflict_rate": self.conflict_rate,
        }
        
        if self.resonant:
            out.update({
                "memory/resonant/final_sim": self.resonant.final_sim,
                "memory/resonant/steps": float(self.resonant.convergence_steps),
                "memory/resonant/entropy": self.resonant.bucket_entropy,
            })
            
        if self.vsa:
            out.update({
                "memory/vsa/novelty": self.vsa.novelty_ema,
                "memory/vsa/rejection_rate": self.vsa.write_rejection_rate,
                "memory/vsa/confidence": self.vsa.match_confidence,
            })
            
        if self.rmf:
            out.update({
                "memory/rmf/field_rms": self.rmf.field_rms,
                "memory/rmf/bias_ratio": self.rmf.bias_ratio,
                "memory/rmf/error": self.rmf.prediction_error,
            })

        for k, v in self.aux.items():
            out[f"memory/aux/{k}"] = v
            
        return out
