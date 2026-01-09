"""Default token collector for language-model style training."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from caramba.config.collector import DefaultCollectorConfig
from caramba.config.train import TrainConfig
from caramba.data.datasets.builder import TokenDatasetBuilder
from caramba.carmath import train_val_counts
from caramba.trainer.upcycle_context import UpcycleContext
from caramba.runtime.tensordict_utils import TensorDictBase, collate_tensordict


def _resolve_data_path(spec: str) -> Path:
    """Resolve a dataset path from a manifest-friendly string.

    Supports:
    - Absolute paths
    - Relative paths resolved from CWD
    - Common convention: `./data/<file>`
    """
    raw = str(spec).strip()
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    # Relative to CWD (caramba is typically invoked from repo root)
    cwd_p = Path.cwd() / p
    if cwd_p.exists():
        return cwd_p

    # Common convention: datasets stored under ./data/
    data_p = Path.cwd() / "data" / p
    if data_p.exists():
        return data_p

    # If nothing found, return original (for error messaging downstream).
    return p


class DefaultCollector:
    def __init__(self, config: DefaultCollectorConfig) -> None:
        self.config = config

    def build_loaders(
        self, train: TrainConfig, ctx: UpcycleContext
    ) -> tuple[DataLoader[TensorDictBase], DataLoader[TensorDictBase] | None]:
        path = _resolve_data_path(str(ctx.group.data))
        if not path.exists():
            tried = [
                str(Path(str(ctx.group.data))),
                str(Path.cwd() / str(ctx.group.data)),
                str(Path.cwd() / "data" / str(ctx.group.data)),
            ]
            raise FileNotFoundError(
                f"Dataset file not found: {ctx.group.data!r}. Tried: {tried}. "
                f"Fix by setting `groups[].data` to an absolute path or placing the file under `./data/`."
            )
        dataset = TokenDatasetBuilder.build(path=path, block_size=int(train.block_size))

        val_frac = float(ctx.defaults.data.val_frac) if ctx.defaults else 0.0
        n = len(cast(Sized, dataset))
        n_train, n_val = train_val_counts(n, val_frac)

        if n_val > 0 and n_train > 0:
            train_ds = Subset(dataset, range(0, n_train))
            val_ds = Subset(dataset, range(n_train, n_train + n_val))
        else:
            train_ds = dataset
            val_ds = None

        batch_size = int(ctx.runtime_plan.batch_size)
        use_pin_memory = bool(train.pin_memory) and ctx.device.type == "cuda"
        loader_kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "drop_last": True,
            "num_workers": int(train.num_workers),
            "pin_memory": use_pin_memory,
        }
        if int(train.num_workers) > 0:
            loader_kwargs["prefetch_factor"] = int(train.prefetch_factor)

        if ctx.dist_ctx is not None:
            train_loader = ctx.dist_ctx.wrap_dataloader(
                train_ds, shuffle=True, **loader_kwargs  # type: ignore[arg-type]
            )
            val_loader = (
                ctx.dist_ctx.wrap_dataloader(
                    val_ds, shuffle=False, **loader_kwargs  # type: ignore[arg-type]
                )
                if val_ds is not None
                else None
            )
            return train_loader, val_loader

        train_loader = DataLoader(train_ds, shuffle=True, collate_fn=collate_tensordict, **loader_kwargs)  # type: ignore[arg-type]
        val_loader = (
            DataLoader(val_ds, shuffle=False, collate_fn=collate_tensordict, **loader_kwargs)  # type: ignore[arg-type]
            if val_ds is not None
            else None
        )
        return train_loader, val_loader

    def next_batch(
        self,
        loader: DataLoader[TensorDictBase],
        iterator: Iterator[TensorDictBase],
    ) -> tuple[TensorDictBase, Iterator[TensorDictBase]]:
        it = iterator
        try:
            return next(it), it
        except StopIteration:
            new_iter = iter(loader)
            return next(new_iter), new_iter

