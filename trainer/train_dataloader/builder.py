"""Standard training DataLoader builder.

This module keeps DataLoader construction logic out of trainer loops so trainers
can focus on orchestration, not dataset splitting/collation/worker settings.
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset

from caramba.carmath import train_val_counts
from caramba.config.defaults import Defaults
from caramba.config.train import TrainConfig
from caramba.runtime.tensordict_utils import TensorDictBase, collate_tensordict


class TrainDataLoaderBuilder:
    """Build a training DataLoader for a dataset component."""

    def build(
        self,
        *,
        dataset_comp: object,
        defaults: Defaults,
        train: TrainConfig,
        device: torch.device,
        batch_size: int,
        dist_ctx: object | None = None,
    ) -> DataLoader[TensorDictBase]:
        if not hasattr(dataset_comp, "build"):
            raise TypeError("Dataset component does not expose build()")
        dataset = dataset_comp.build()  # type: ignore[attr-defined]

        val_frac = float(defaults.data.val_frac)
        n = len(cast(Sized, dataset))
        n_train, _n_val = train_val_counts(n, float(val_frac))
        train_ds = Subset(dataset, range(0, n_train))

        loader_kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
            "num_workers": int(train.num_workers),
            "pin_memory": bool(train.pin_memory) and device.type == "cuda",
            "drop_last": True,
            "collate_fn": collate_tensordict,
        }
        if int(train.num_workers) > 0:
            loader_kwargs["prefetch_factor"] = int(getattr(train, "prefetch_factor", 2))
            loader_kwargs["persistent_workers"] = True

        if dist_ctx is not None and hasattr(dist_ctx, "wrap_dataloader"):
            return dist_ctx.wrap_dataloader(train_ds, shuffle=True, **loader_kwargs)  # type: ignore[no-any-return, attr-defined]
        return DataLoader(train_ds, shuffle=True, **loader_kwargs)

