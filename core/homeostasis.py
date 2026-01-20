"""Intrinsic drives and homeostatic control primitives.

These are *framework-level* building blocks for the "organism" architecture:
- agents (or models) expose internal metrics (entropy, energy, alignment, etc.)
- a homeostatic controller turns metric deviations into impulse events

This module is intentionally runtime-agnostic: it does not depend on brainstorm
or any specific agent loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.event import EventEnvelope


@dataclass(frozen=True, slots=True)
class DriveBand:
    """A healthy operating band [min_value, max_value]."""

    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        lo = float(self.min_value)
        hi = float(self.max_value)
        if lo > hi:
            raise ValueError(f"DriveBand requires min_value <= max_value, got {lo} > {hi}")

    def deviation(self, value: float) -> float:
        v = float(value)
        lo = float(self.min_value)
        hi = float(self.max_value)
        if v < lo:
            return float(lo - v)
        if v > hi:
            return float(v - hi)
        return 0.0


@dataclass(frozen=True, slots=True)
class DriveSignal:
    """Evaluated drive signal for a single metric."""

    name: str
    metric: str
    value: float
    band: DriveBand
    deviation: float
    urgency: float

    def to_json(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "metric": str(self.metric),
            "value": float(self.value),
            "band": {"min": float(self.band.min_value), "max": float(self.band.max_value)},
            "deviation": float(self.deviation),
            "urgency": float(self.urgency),
        }


@dataclass(frozen=True, slots=True)
class IntrinsicDrive:
    """A single intrinsic drive specification."""

    name: str
    metric: str
    band: DriveBand
    weight: float = 1.0

    def evaluate(self, metrics: Mapping[str, float]) -> DriveSignal:
        if self.metric not in metrics:
            raise KeyError(f"Missing metric {self.metric!r} required by drive {self.name!r}")
        v = float(metrics[self.metric])
        dev = float(self.band.deviation(v))
        urg = float(self.weight) * dev
        return DriveSignal(
            name=str(self.name),
            metric=str(self.metric),
            value=float(v),
            band=self.band,
            deviation=float(dev),
            urgency=float(urg),
        )


@dataclass(frozen=True, slots=True)
class HomeostaticLoop:
    """Turn metric deviations into impulse events."""

    drives: tuple[IntrinsicDrive, ...]
    sender: str
    # Emit an impulse only if max urgency exceeds this threshold.
    impulse_threshold: float = 0.0
    # Optional per-impulse budget suggestion.
    budget_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.drives:
            raise ValueError("HomeostaticLoop requires at least one drive")
        if not isinstance(self.sender, str) or not self.sender.strip():
            raise ValueError("HomeostaticLoop.sender must be a non-empty string")
        thr = float(self.impulse_threshold)
        if thr < 0.0:
            raise ValueError(f"impulse_threshold must be >= 0, got {thr}")
        if self.budget_ms is not None and int(self.budget_ms) < 0:
            raise ValueError(f"budget_ms must be >= 0, got {self.budget_ms}")

    def evaluate(self, metrics: Mapping[str, float]) -> tuple[DriveSignal, ...]:
        return tuple(d.evaluate(metrics) for d in self.drives)

    def impulse(self, metrics: Mapping[str, float]) -> EventEnvelope | None:
        signals = self.evaluate(metrics)
        max_urg = max(float(s.urgency) for s in signals)
        if max_urg <= float(self.impulse_threshold):
            return None

        # Priority is proportional to urgency; keep it integer for envelopes.
        priority = round(max_urg * 1000.0)
        payload = {
            "metrics": {str(k): float(v) for k, v in metrics.items()},
            "signals": [s.to_json() for s in signals],
            "max_urgency": float(max_urg),
        }
        return EventEnvelope(
            type="Impulse",
            payload=payload,
            sender=str(self.sender),
            priority=int(priority),
            budget_ms=int(self.budget_ms) if self.budget_ms is not None else None,
        )

