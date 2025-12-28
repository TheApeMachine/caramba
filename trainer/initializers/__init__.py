"""Initializer components for Upcycle."""

from __future__ import annotations

from typing import Protocol

from torch import nn

from config.train import TrainConfig
from trainer.upcycle_init_context import UpcycleInitContext


class Initializer(Protocol):
    def init_models(self, train: TrainConfig, ctx: UpcycleInitContext) -> tuple[nn.Module, nn.Module]: ...


__all__ = ["Initializer"]

