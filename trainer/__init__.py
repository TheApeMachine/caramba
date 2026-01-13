"""Training loop components for model training and upcycling.

Training in caramba has two main modes:
1. Standard training: Train a model from scratch or fine-tune
2. Upcycling: Convert a pretrained model to a new architecture (like DBA)

This package provides:
- Trainer: Main training loop orchestrator (single entrypoint)
- Upcycle internals: Teacher-student distillation for architecture conversion
- Blockwise internals: Layer-by-layer distillation for stable upcycling
- DistillLoss: L1 loss for knowledge transfer
- Distributed: DDP/FSDP support for multi-GPU training
"""
from __future__ import annotations

import importlib
from typing import Any

exports: dict[str, tuple[str, str]] = {
    "BlockwiseTrainer": ("caramba.trainer.blockwise", "BlockwiseTrainer"),
    "DistillLoss": ("caramba.trainer.distill", "DistillLoss"),
    "Trainer": ("caramba.trainer.trainer", "Trainer"),
    "StandardTrainer": ("caramba.trainer.trainer", "Trainer"),
    "UpcycleTrainer": ("caramba.trainer.upcycle", "UpcycleTrainer"),
    "DistributedStrategy": ("caramba.trainer.distributed", "DistributedStrategy"),
    "DistributedConfig": ("caramba.trainer.distributed", "DistributedConfig"),
    "DistributedContext": ("caramba.trainer.distributed", "DistributedContext"),
    "is_distributed": ("caramba.trainer.distributed", "is_distributed"),
    "get_rank": ("caramba.trainer.distributed", "get_rank"),
    "get_world_size": ("caramba.trainer.distributed", "get_world_size"),
    "is_main_process": ("caramba.trainer.distributed", "is_main_process"),
}

__all__ = [
    "BlockwiseTrainer",
    "DistillLoss",
    "Trainer",
    "StandardTrainer",
    "UpcycleTrainer",
    "DistributedStrategy",
    "DistributedConfig",
    "DistributedContext",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
]


def __getattr__(name: str) -> Any:
    """Lazily import trainer symbols.

    The trainer package has a wide dependency surface area. Keeping imports
    lazy prevents unrelated modules (e.g. steppers) from failing import because
    of optional components in other subpackages.
    """
    if name not in exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = exports[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __dir__() -> list[str]:
    """Return a stable list of exported names for auto-completion."""
    return sorted(set(list(globals().keys()) + list(exports.keys())))
