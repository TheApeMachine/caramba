"""Collector components for Upcycle/Standard training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from torch.utils.data import DataLoader

from config.train import TrainConfig
from trainer.upcycle_context import UpcycleContext
from runtime.tensordict_utils import TensorDictBase


class Collector(Protocol):
    def build_loaders(
        self, train: TrainConfig, ctx: UpcycleContext
    ) -> tuple[DataLoader[TensorDictBase], DataLoader[TensorDictBase] | None]: ...

    def next_batch(
        self,
        loader: DataLoader[TensorDictBase],
        iterator: Iterator[TensorDictBase],
    ) -> tuple[TensorDictBase, Iterator[TensorDictBase]]: ...


__all__ = ["Collector"]

