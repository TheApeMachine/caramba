"""Swap/offload manager (UMA-aware policies).

This is a small abstraction for:
- optimizer state offload/reload
- (future) activation checkpoint policy selection

This component controls optimizer state staging/offload policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SwapManager:
    """Optimizer state staging."""

    offload_optimizer: bool = False
    offload_device: str = "cpu"

    def before_optimizer_step(self, optimizer: torch.optim.Optimizer, *, device: torch.device) -> None:
        if not self.offload_optimizer:
            return
        try:
            from caramba.optimizer.offload import load_optimizer_state

            load_optimizer_state(optimizer, device=device)
        except Exception as e:
            from caramba.console import logger as console_logger
            console_logger.warning(f"SwapManager: Failed to load optimizer state: {e}")

    def after_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.offload_optimizer:
            return
        try:
            from caramba.optimizer.offload import offload_optimizer_state

            offload_optimizer_state(optimizer, device=torch.device(self.offload_device))
        except Exception as e:
            from caramba.console import logger as console_logger
            console_logger.warning(f"SwapManager: Failed to offload optimizer state: {e}")

