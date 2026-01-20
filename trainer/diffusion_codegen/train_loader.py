"""DataLoader builder for diffusion codegen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from config.train import TrainConfig
from runtime.tensordict_utils import collate_tensordict


@dataclass(frozen=True, slots=True)
class LoaderFactory:
    """Build a DataLoader for a dataset component."""

    def build(self, *, dataset_comp: object, train: TrainConfig, device: torch.device) -> DataLoader:
        if not hasattr(dataset_comp, "build"):
            raise TypeError("Dataset component does not expose build()")
        dataset = dataset_comp.build()  # type: ignore[attr-defined]

        pin = bool(getattr(train, "pin_memory", False)) and device.type == "cuda"
        kwargs: dict[str, Any] = {
            "batch_size": int(train.batch_size),
            "shuffle": True,
            "drop_last": True,
            "num_workers": int(getattr(train, "num_workers", 0)),
            "pin_memory": bool(pin),
            "collate_fn": lambda items: collate_tensordict(items),
        }
        if int(kwargs["num_workers"]) > 0:
            kwargs["prefetch_factor"] = int(getattr(train, "prefetch_factor", 2))
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

