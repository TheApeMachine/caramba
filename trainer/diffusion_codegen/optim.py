"""Optimization utilities for diffusion codegen

Builds optimizers/schedulers from TrainConfig for the diffusion codegen recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from caramba.config.train import TrainConfig


@dataclass(frozen=True, slots=True)
class OptimizerFactory:
    """Optimizer and scheduler builder."""

    def buildOptimizer(self, *, model: nn.Module, train: TrainConfig) -> Optimizer:
        kind = str(getattr(train, "optimizer", "adamw")).lower().strip()
        lr = float(train.lr)
        wd = float(getattr(train, "weight_decay", 0.0))
        if kind == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if kind == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        raise ValueError(f"Unsupported optimizer for diffusion_codegen: {kind!r}")

    def buildScheduler(self, *, optimizer: Optimizer, train: TrainConfig, steps: int) -> CosineAnnealingLR | None:
        kind = str(getattr(train, "scheduler", "none")).lower().strip()
        if kind in {"none", ""}:
            return None
        if kind == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, int(steps)))
        raise ValueError(f"Unsupported scheduler for diffusion_codegen: {kind!r}")

