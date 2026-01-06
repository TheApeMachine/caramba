"""Exponential moving average (EMA)

Tracks a moving average of model parameters for more stable sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ExponentialMovingAverage:
    """EMA tracker

    Maintains shadow weights and supports applying/restoring them to a model.
    """

    decay: float = 0.999

    def __post_init__(self) -> None:
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    def register(self, *, model: nn.Module) -> None:
        """Initialize EMA buffers from model parameters."""

        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, *, model: nn.Module) -> None:
        """Update EMA buffers from current model parameters."""

        if not self.shadow:
            raise RuntimeError("EMA has not been registered. Call register(model=...) first.")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                raise KeyError(f"EMA shadow is missing parameter: {name}")
            self.shadow[name].mul_(float(self.decay)).add_(param.detach(), alpha=1.0 - float(self.decay))

    def apply(self, *, model: nn.Module) -> None:
        """Apply EMA weights to model (with restore support)."""

        if not self.shadow:
            raise RuntimeError("EMA has not been registered. Call register(model=...) first.")
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                raise KeyError(f"EMA shadow is missing parameter: {name}")
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])  # type: ignore[call-arg]

    def restore(self, *, model: nn.Module) -> None:
        """Restore original weights after apply()."""

        if not self.backup:
            raise RuntimeError("EMA restore requested but no backup exists (did you call apply()?)")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.backup:
                raise KeyError(f"EMA backup is missing parameter: {name}")
            param.data.copy_(self.backup[name])  # type: ignore[call-arg]
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        """Serialize EMA state."""

        return {"decay": float(self.decay), "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load EMA state."""

        if "decay" in state:
            self.decay = float(state["decay"])
        shadow = state.get("shadow", None)
        if not isinstance(shadow, dict):
            raise TypeError("EMA state_dict must include a dict 'shadow'.")
        self.shadow = {str(k): v.detach().clone() for k, v in shadow.items()}

