"""TensorDict utilities.

Caramba treats `tensordict.TensorDict` as the canonical batch/output container.
We still accept plain python containers at boundaries (datasets, external code),
but convert aggressively to TensorDict inside training/evaluation loops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, cast

import torch
from torch import Tensor

from tensordict import TensorDict
from tensordict.base import TensorDictBase

__all__ = [
    "TensorDictBase",
    "as_tensordict",
    "collate_tensordict",
    "to_device",
    "tree_map",
]


def _infer_batch_size(d: Mapping[str, Any]) -> list[int]:
    """Best-effort batch size inference for TensorDict construction."""
    b0: int | None = None
    for v in d.values():
        if isinstance(v, Tensor) and v.dim() >= 1:
            b = int(v.shape[0])
            if b0 is None:
                b0 = b
            elif b0 != b:
                return []
    return [] if b0 is None else [int(b0)]


def as_tensordict(obj: Any) -> "TensorDictBase":
    """Convert a supported container into a TensorDictBase.

    Supported inputs:
    - TensorDictBase: returned as-is
    - Mapping[str, Any]: recursively converted; leaves are tensors
    """
    if isinstance(obj, TensorDictBase):  # type: ignore[arg-type]
        return cast("TensorDictBase", obj)
    if not isinstance(obj, Mapping):
        raise TypeError(f"Expected TensorDictBase or Mapping, got {type(obj).__name__}")

    # Recursively convert nested mappings to nested TensorDicts.
    payload: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, Mapping):
            payload[str(k)] = as_tensordict(v)
        else:
            payload[str(k)] = v
    bs = _infer_batch_size(payload)
    return TensorDict(payload, batch_size=bs)


def tree_map(fn: Callable[[Any], Any], obj: Any) -> Any:
    """Apply `fn` to all leaves of a nested container."""
    if isinstance(obj, Tensor):
        return fn(obj)
    if isinstance(obj, Mapping):
        return {k: tree_map(fn, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_map(fn, v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tree_map(fn, v) for v in obj)
    return fn(obj)


def to_device(td: Any, *, device: torch.device, non_blocking: bool = False) -> Any:
    """Move tensors (and TensorDicts) onto `device`."""

    def move(x: Any) -> Any:
        if isinstance(x, Tensor):
            return x.to(device=device, non_blocking=bool(non_blocking))
        if isinstance(x, TensorDictBase):  # type: ignore[arg-type]
            return x.to(device=device, non_blocking=bool(non_blocking))
        return x

    return tree_map(move, td)


def collate_tensordict(items: list[Any]) -> TensorDictBase:
    """Collate a list of samples into a batch TensorDict.

    - If samples are TensorDicts, uses `TensorDict.stack` (fast, correct).
    - Otherwise, falls back to PyTorch `default_collate` and converts to TensorDict.
    """
    if not items:
        raise ValueError("collate_tensordict: empty batch")
    first = items[0]
    if isinstance(first, TensorDictBase):  # type: ignore[arg-type]
        if not all(isinstance(it, TensorDictBase) for it in items):  # type: ignore[arg-type]
            types = ", ".join(sorted({type(it).__name__ for it in items}))
            raise TypeError(
                "collate_tensordict: mixed batch types; expected all items to be TensorDictBase "
                f"when the first item is TensorDictBase (got: {types})"
            )
        # `TensorDict.stack` handles nested tensordicts as well.
        return TensorDict.stack(items, 0)  # type: ignore[arg-type]
    from torch.utils.data._utils.collate import default_collate

    return as_tensordict(default_collate(items))

