"""Training-time collectors/hooks.

These hooks are used by training steppers to keep the trainer orchestration thin
and make cross-cutting concerns (logging, checkpoints, layer stats, etc.)
composable.
"""

from __future__ import annotations

from caramba.trainer.collector.training.hooks import TrainHook

__all__ = ["TrainHook"]
