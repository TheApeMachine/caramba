"""Training telemetry for orchestrator decision-making.

The telemetry stream provides real-time diagnostics that the orchestrator
uses to decide when and how to switch strategies:

- Loss trajectory: smoothed loss, slope, variance, best seen
- Gradient health: norms (global, per-layer), spike detection
- Curvature proxies: optional sharpness estimates
- Training phase: early/mid/late based on step count

The key insight is that different strategies excel at different regimes:
- High variance → need stability (AdaGC, lower LR)
- Plateauing → need escape (SGD momentum, LR bump)
- Spikes → need clipping/reset
- Smooth convergence → can be aggressive
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    pass


class TrainingPhase(str, Enum):
    """Coarse training phase for phase-aware strategy selection."""

    EARLY = "early"  # First ~10% of training
    WARMUP = "warmup"  # During warmup period
    MID = "mid"  # Middle ~60%
    LATE = "late"  # Final ~30%
    CONVERGING = "converging"  # Loss decreasing steadily
    PLATEAU = "plateau"  # Loss stuck
    UNSTABLE = "unstable"  # High variance or spikes


@dataclass
class TelemetrySnapshot:
    """A single telemetry observation at one step.

    This is the "state" that the bandit/orchestrator observes to make
    strategy selection decisions.
    """

    step: int = 0

    # Loss metrics (smoothed and raw)
    loss: float = 0.0
    loss_ema: float = 0.0
    loss_variance: float = 0.0
    loss_slope: float = 0.0  # Derivative of EMA
    best_loss: float = float("inf")
    loss_improvement: float = 0.0  # best_loss - current

    # Gradient metrics
    grad_norm: float = 0.0
    grad_norm_ema: float = 0.0
    grad_norm_variance: float = 0.0
    max_layer_grad_norm: float = 0.0
    min_layer_grad_norm: float = 0.0
    grad_norm_ratio: float = 1.0  # max/min, indicates imbalance

    # Spike detection
    spike_count: int = 0  # Number of spikes in recent window
    spike_score: float = 0.0  # Weighted spike severity
    last_spike_steps_ago: int = -1  # -1 means no recent spike

    # Update metrics
    update_norm: float = 0.0
    update_norm_ema: float = 0.0
    param_update_ratio: float = 0.0  # ||Δθ|| / ||θ||

    # LR and optimizer state
    lr: float = 0.0
    momentum_estimate: float = 0.0

    # Phase estimation
    phase: TrainingPhase = TrainingPhase.EARLY
    phase_confidence: float = 0.0

    # Optional curvature proxy (expensive to compute)
    sharpness: float | None = None

    # Arbitrary user-provided metrics (for metric-agnostic orchestrator use).
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "step": self.step,
            "loss": self.loss,
            "loss_ema": self.loss_ema,
            "loss_variance": self.loss_variance,
            "loss_slope": self.loss_slope,
            "best_loss": self.best_loss,
            "grad_norm": self.grad_norm,
            "grad_norm_ema": self.grad_norm_ema,
            "spike_count": self.spike_count,
            "spike_score": self.spike_score,
            "phase": self.phase.value,
            "lr": self.lr,
            "metrics": dict(self.metrics),
        }

    def is_healthy(self) -> bool:
        """Quick check for obviously unhealthy training state."""
        if self.spike_count > 3:
            return False
        if self.loss_variance > 0.5 * self.loss_ema:
            return False
        if self.phase == TrainingPhase.UNSTABLE:
            return False
        return True


class SpikeDetector:
    """Detects loss/gradient spikes using EMA-based thresholds.

    A spike is defined as a value that exceeds the EMA by more than
    a configurable number of standard deviations.

    This is inspired by AdaGC's approach but generalized to track
    any metric.
    """

    def __init__(
        self,
        *,
        ema_decay: float = 0.99,
        threshold_std: float = 3.0,
        window_size: int = 100,
    ) -> None:
        """Initialize spike detector.

        Args:
            ema_decay: Decay factor for EMA (higher = smoother).
            threshold_std: Number of std devs to trigger spike.
            window_size: Window for spike counting.
        """
        self.ema_decay = ema_decay
        self.threshold_std = threshold_std
        self.window_size = window_size

        self._ema: float | None = None
        self._ema_sq: float | None = None  # For variance
        self._spike_history: deque[int] = deque(maxlen=window_size)
        self._step = 0

    @property
    def ema(self) -> float:
        """Current EMA value."""
        return self._ema if self._ema is not None else 0.0

    @property
    def variance(self) -> float:
        """Current variance estimate."""
        if self._ema is None or self._ema_sq is None:
            return 0.0
        var = self._ema_sq - self._ema**2
        return max(0.0, var)

    @property
    def std(self) -> float:
        """Current standard deviation estimate."""
        return self.variance**0.5

    @property
    def threshold(self) -> float:
        """Current spike threshold."""
        return self.ema + self.threshold_std * self.std

    def update(self, value: float) -> bool:
        """Update with new value, return True if spike detected.

        Args:
            value: New observation.

        Returns:
            True if this value is a spike.
        """
        self._step += 1

        # Initialize EMAs on first observation
        if self._ema is None:
            self._ema = value
            self._ema_sq = value**2
            self._spike_history.append(0)
            return False

        # Check for spike before updating EMA
        is_spike = value > self.threshold

        # Update EMAs (we know _ema and _ema_sq are not None here due to the check above)
        ema = self._ema  # Guaranteed not None after the check above
        ema_sq = self._ema_sq  # Guaranteed not None
        assert ema is not None and ema_sq is not None
        self._ema = self.ema_decay * ema + (1 - self.ema_decay) * value
        self._ema_sq = self.ema_decay * ema_sq + (1 - self.ema_decay) * value**2

        # Track spike history
        self._spike_history.append(1 if is_spike else 0)

        return is_spike

    @property
    def spike_count(self) -> int:
        """Number of spikes in recent window."""
        return sum(self._spike_history)

    @property
    def spike_score(self) -> float:
        """Weighted spike score (recent spikes weighted higher)."""
        if not self._spike_history:
            return 0.0

        total = 0.0
        weight = 1.0
        decay = 0.9
        for spike in reversed(self._spike_history):
            total += spike * weight
            weight *= decay

        return total

    def reset(self) -> None:
        """Reset the detector state."""
        self._ema = None
        self._ema_sq = None
        self._spike_history.clear()
        self._step = 0


class TelemetryStream:
    """Continuous telemetry stream for training monitoring.

    Collects and processes training metrics to produce TelemetrySnapshots
    that the orchestrator uses for decision-making.
    """

    def __init__(
        self,
        *,
        ema_decay: float = 0.99,
        window_size: int = 100,
        total_steps: int = 10000,
        warmup_steps: int = 0,
    ) -> None:
        """Initialize telemetry stream.

        Args:
            ema_decay: Decay for EMAs.
            window_size: Window for variance/spike tracking.
            total_steps: Expected total training steps.
            warmup_steps: Number of warmup steps.
        """
        self.ema_decay = ema_decay
        self.window_size = window_size
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        # Core detectors
        self._loss_detector = SpikeDetector(
            ema_decay=ema_decay,
            threshold_std=3.0,
            window_size=window_size,
        )
        self._grad_detector = SpikeDetector(
            ema_decay=ema_decay,
            threshold_std=4.0,  # Grad spikes need higher threshold
            window_size=window_size,
        )

        # History for slope computation
        self._loss_history: deque[float] = deque(maxlen=window_size)
        self._grad_history: deque[float] = deque(maxlen=window_size)

        # Best tracking
        self._best_loss = float("inf")
        self._step = 0
        self._last_spike_step = -1

        # Layer-wise grad norms
        self._layer_grad_norms: dict[str, float] = {}

        # Update metrics
        self._update_norm_ema: float = 0.0
        self._param_norm: float = 1.0

    def record(
        self,
        *,
        loss: float,
        grad_norm: float,
        lr: float,
        model: nn.Module | None = None,
        update_norm: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> TelemetrySnapshot:
        """Record one training step and return a snapshot.

        Args:
            loss: Training loss.
            grad_norm: Global gradient norm.
            lr: Current learning rate.
            model: Optional model for layer-wise stats.
            update_norm: Optional update norm (||Δθ||).

        Returns:
            TelemetrySnapshot with current state.
        """
        self._step += 1

        # Update detectors
        loss_spike = self._loss_detector.update(loss)
        grad_spike = self._grad_detector.update(grad_norm)

        if loss_spike or grad_spike:
            self._last_spike_step = self._step

        # Track history
        self._loss_history.append(loss)
        self._grad_history.append(grad_norm)

        # Best loss tracking
        if loss < self._best_loss:
            self._best_loss = loss

        # Compute loss slope
        loss_slope = self._compute_slope(self._loss_history)

        # Layer-wise grad norms (if model provided)
        if model is not None:
            self._update_layer_grad_norms(model)

        # Update norm tracking
        if update_norm is not None:
            self._update_norm_ema = (
                self.ema_decay * self._update_norm_ema
                + (1 - self.ema_decay) * update_norm
            )

        # Param norm for ratio
        if model is not None and update_norm is not None:
            self._param_norm = self._compute_param_norm(model)

        # Determine phase
        phase, confidence = self._determine_phase(loss_slope)

        # Build snapshot
        snapshot = TelemetrySnapshot(
            step=self._step,
            loss=loss,
            loss_ema=self._loss_detector.ema,
            loss_variance=self._loss_detector.variance,
            loss_slope=loss_slope,
            best_loss=self._best_loss,
            loss_improvement=self._best_loss - loss,
            grad_norm=grad_norm,
            grad_norm_ema=self._grad_detector.ema,
            grad_norm_variance=self._grad_detector.variance,
            max_layer_grad_norm=max(self._layer_grad_norms.values(), default=0.0),
            min_layer_grad_norm=min(
                (v for v in self._layer_grad_norms.values() if v > 0),
                default=0.0,
            ),
            grad_norm_ratio=self._compute_grad_ratio(),
            spike_count=self._loss_detector.spike_count + self._grad_detector.spike_count,
            spike_score=self._loss_detector.spike_score + self._grad_detector.spike_score,
            last_spike_steps_ago=(
                self._step - self._last_spike_step
                if self._last_spike_step > 0
                else -1
            ),
            update_norm=update_norm or 0.0,
            update_norm_ema=self._update_norm_ema,
            param_update_ratio=(
                self._update_norm_ema / self._param_norm
                if self._param_norm > 0
                else 0.0
            ),
            lr=lr,
            phase=phase,
            phase_confidence=confidence,
            metrics={} if metrics is None else {str(k): float(v) for k, v in metrics.items()},
        )

        return snapshot

    def _compute_slope(self, history: deque[float]) -> float:
        """Compute slope of values in history using linear regression."""
        if len(history) < 5:
            return 0.0

        # Simple linear regression slope
        n = len(history)
        x_mean = (n - 1) / 2
        y_mean = sum(history) / n

        numerator = 0.0
        denominator = 0.0
        for i, y in enumerate(history):
            numerator += (i - x_mean) * (y - y_mean)
            denominator += (i - x_mean) ** 2

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _update_layer_grad_norms(self, model: nn.Module) -> None:
        """Compute per-layer gradient norms."""
        self._layer_grad_norms.clear()
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = float(param.grad.data.norm(2).item())
                # Group by layer (first part of name)
                layer_name = name.split(".")[0]
                if layer_name in self._layer_grad_norms:
                    self._layer_grad_norms[layer_name] = max(
                        self._layer_grad_norms[layer_name], norm
                    )
                else:
                    self._layer_grad_norms[layer_name] = norm

    def _compute_param_norm(self, model: nn.Module) -> float:
        """Compute total parameter norm."""
        total = 0.0
        for param in model.parameters():
            total += float(param.data.norm(2).item() ** 2)
        return total**0.5

    def _compute_grad_ratio(self) -> float:
        """Compute max/min layer gradient ratio."""
        if not self._layer_grad_norms:
            return 1.0

        vals = list(self._layer_grad_norms.values())
        max_norm = max(vals)
        # If all grads are ~0 (can happen early with fp16 / small losses), avoid crashing.
        positives = [v for v in vals if v > 1e-10]
        if not positives:
            return 1.0
        min_norm = min(positives)

        if min_norm < 1e-10:
            return 1.0

        return max_norm / min_norm

    def _determine_phase(self, loss_slope: float) -> tuple[TrainingPhase, float]:
        """Determine current training phase.

        Returns:
            Tuple of (phase, confidence).
        """
        # Check for instability first
        if self._loss_detector.spike_count > 2 or self._grad_detector.spike_count > 3:
            return TrainingPhase.UNSTABLE, 0.9

        # Check warmup
        if self._step <= self.warmup_steps:
            return TrainingPhase.WARMUP, 0.95

        # Position in training
        progress = self._step / max(1, self.total_steps)

        if progress < 0.1:
            phase = TrainingPhase.EARLY
        elif progress > 0.7:
            phase = TrainingPhase.LATE
        else:
            phase = TrainingPhase.MID

        # Override with trajectory-based phase
        if loss_slope < -1e-5:  # Decreasing
            return TrainingPhase.CONVERGING, 0.8
        elif abs(loss_slope) < 1e-6:  # Flat
            return TrainingPhase.PLATEAU, 0.7

        return phase, 0.6

    def get_context_vector(self) -> list[float]:
        """Get a fixed-size context vector for bandit algorithms.

        This encodes the current training state in a form suitable for
        contextual bandits.
        """
        return [
            self._loss_detector.ema,
            self._loss_detector.variance,
            self._grad_detector.ema,
            self._grad_detector.variance,
            float(self._loss_detector.spike_count) / self.window_size,
            float(self._grad_detector.spike_count) / self.window_size,
            self._step / max(1, self.total_steps),  # Progress
            float(self._step <= self.warmup_steps),  # Is warmup
        ]

    def reset(self) -> None:
        """Reset telemetry state."""
        self._loss_detector.reset()
        self._grad_detector.reset()
        self._loss_history.clear()
        self._grad_history.clear()
        self._best_loss = float("inf")
        self._step = 0
        self._last_spike_step = -1
        self._layer_grad_norms.clear()
        self._update_norm_ema = 0.0
