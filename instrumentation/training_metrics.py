"""Lightweight training metrics for memory tuner.

This provides a zero-overhead way to pass loss/accuracy from the trainer to memory blocks
without threading ctx through the forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass
class TrainingMetrics:
    """Thread-safe container for current training step metrics."""
    
    _lock: Lock
    _loss: float | None = None
    _accuracy: float | None = None
    _step: int = 0
    
    def __init__(self) -> None:
        object.__setattr__(self, '_lock', Lock())
        object.__setattr__(self, '_loss', None)
        object.__setattr__(self, '_accuracy', None)
        object.__setattr__(self, '_step', 0)
    
    def update(self, *, step: int, loss: float | None = None, accuracy: float | None = None) -> None:
        """Update metrics (called by trainer after each step)."""
        with self._lock:
            object.__setattr__(self, '_step', step)
            if loss is not None:
                object.__setattr__(self, '_loss', loss)
            if accuracy is not None:
                object.__setattr__(self, '_accuracy', accuracy)
    
    @property
    def loss(self) -> float | None:
        with self._lock:
            return self._loss
    
    @property
    def accuracy(self) -> float | None:
        with self._lock:
            return self._accuracy
    
    @property
    def step(self) -> int:
        with self._lock:
            return self._step


# Global singleton - zero overhead access
_metrics = TrainingMetrics()


def get_training_metrics() -> TrainingMetrics:
    """Get the global training metrics singleton."""
    return _metrics


def update_training_metrics(*, step: int, loss: float | None = None, accuracy: float | None = None) -> None:
    """Update the global training metrics (called by trainer)."""
    _metrics.update(step=step, loss=loss, accuracy=accuracy)
