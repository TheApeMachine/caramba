"""Cast dtype transform

Converts tensors to specific dtypes, ensuring model inputs match expected
types without modifying the original dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from runtime.tensordict_utils import TensorDictBase, as_tensordict


def _dtype_from_str(s: str) -> torch.dtype:
    """Convert dtype string to torch dtype

    Parses common dtype name variations (like "fp32" or "float32") into PyTorch
    dtype objects, making config files more readable than requiring enum values.
    """
    t = str(s).lower()
    if t in ("float32", "fp32"):
        return torch.float32
    if t in ("float16", "fp16", "half"):
        return torch.float16
    if t in ("bfloat16", "bf16"):
        return torch.bfloat16
    if t in ("int64", "long"):
        return torch.int64
    if t in ("int32",):
        return torch.int32
    if t in ("bool",):
        return torch.bool
    raise ValueError(f"Unknown dtype {s!r}")


@dataclass(frozen=True, slots=True)
class CastDtype:
    """Cast tensor dtypes

    Converts tensors to specific dtypes, essential for ensuring model inputs
    match expected types (e.g., float32 vs float16) without modifying datasets.
    """
    dtypes: dict[str, torch.dtype]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Apply dtype casting

        Converts specified tensors to target dtypes, leaving other keys
        unchanged. This is typically done right before model input to ensure
        type compatibility.
        """
        d = dict(td)
        for k, dt in self.dtypes.items():
            v = d.get(k, None)
            if isinstance(v, Tensor):
                d[k] = v.to(dtype=dt)
        return as_tensordict(d)
