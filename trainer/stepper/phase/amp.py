"""Automatic mixed precision policy for the stepwise training loop."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch
from torch import Tensor
from torch.optim import Optimizer

from caramba.carmath import autocast_dtype
from caramba.console import logger


class AmpController:
    """AMP controller

    This object encapsulates all mixed-precision policy decisions:
    - whether autocast is enabled
    - which dtype autocast uses
    - whether GradScaler is active (CUDA fp16 only)
    """

    def __init__(self, *, train: object, device: torch.device) -> None:
        """Create an AMP controller from a training config and device."""
        self.enabled = bool(getattr(train, "use_amp", False)) and device.type in {"cuda", "mps", "cpu"}
        self.device = device
        self.dtype = autocast_dtype(device, str(getattr(train, "amp_dtype", "auto")))

        if self.enabled and device.type == "mps" and self.dtype == torch.float16:
            logger.warning("MPS fp16 autocast is unstable without GradScaler; switching to bfloat16 autocast.")
            self.dtype = torch.bfloat16

        self.scaler: torch.cuda.amp.GradScaler | None = None
        if self.enabled and device.type == "cuda" and self.dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def context(self) -> AbstractContextManager[object]:
        """Return the torch.autocast context for this controller."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=bool(self.enabled),
        )

    def backward(self, *, loss: Tensor) -> None:
        """Backpropagate `loss`, scaling when GradScaler is active."""
        if self.scaler is None:
            loss.backward()
            return
        self.scaler.scale(loss).backward()

    def unscale(self, *, optimizer: Optimizer) -> None:
        """Unscale gradients for clipping when GradScaler is active."""
        if self.scaler is None:
            return
        self.scaler.unscale_(optimizer)

    def step(self, *, optimizer: Optimizer) -> None:
        """Step the optimizer (and GradScaler if active)."""
        if self.scaler is None:
            optimizer.step()
            return
        self.scaler.step(optimizer)
        self.scaler.update()

