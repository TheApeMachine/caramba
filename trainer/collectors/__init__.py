"""Collector components for Upcycle/Standard training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from torch import Tensor
from torch.utils.data import DataLoader

from config.train import TrainConfig
from trainer.upcycle_context import UpcycleContext


class Collector(Protocol):
    def build_loaders(
        self, train: TrainConfig, ctx: UpcycleContext
    ) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]] | None]: ...

    def next_batch(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        iterator: Iterator[tuple[Tensor, Tensor]],
    ) -> tuple[tuple[Tensor, Tensor], Iterator[tuple[Tensor, Tensor]]]: ...


__all__ = ["Collector"]

