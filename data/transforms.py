"""TensorDict transforms (dict-in/dict-out).

These transforms are intentionally small and composable. They operate on
TensorDicts (or dict-like mappings) and return TensorDicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from caramba.config.transforms import (
    AddMaskTransformConfig,
    CastDtypeTransformConfig,
    GaussianNoiseTransformConfig,
    GraphBatchTransformConfig,
    RenameKeysTransformConfig,
    TokenShiftTransformConfig,
    TransformConfig,
)
from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict


class Transform(Protocol):
    def __call__(self, td: TensorDictBase) -> TensorDictBase: ...


def _dtype_from_str(s: str) -> torch.dtype:
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
class RenameKeys:
    mapping: dict[str, str]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        d = dict(td)
        for src, dst in self.mapping.items():
            if src in d:
                d[dst] = d.pop(src)
        return as_tensordict(d)


@dataclass(frozen=True, slots=True)
class CastDtype:
    dtypes: dict[str, torch.dtype]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        d = dict(td)
        for k, dt in self.dtypes.items():
            v = d.get(k, None)
            if isinstance(v, Tensor):
                d[k] = v.to(dtype=dt)
        return as_tensordict(d)


@dataclass(frozen=True, slots=True)
class AddMask:
    src_key: str
    mask_key: str
    ignore_index: int

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        d = dict(td)
        src = d.get(self.src_key, None)
        if not isinstance(src, Tensor):
            return as_tensordict(d)
        d[self.mask_key] = (src != self.ignore_index)
        return as_tensordict(d)


@dataclass(frozen=True, slots=True)
class TokenShift:
    src_key: str
    input_key: str
    target_key: str

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        d = dict(td)
        tok = d.get(self.src_key, None)
        if not isinstance(tok, Tensor):
            return as_tensordict(d)
        if tok.dim() < 1 or tok.size(-1) < 2:
            raise ValueError(f"token_shift expects {self.src_key} with last dim >= 2")
        d[self.input_key] = tok[..., :-1]
        d[self.target_key] = tok[..., 1:]
        return as_tensordict(d)


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    src_key: str
    out_key: str
    sigma: float

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        d = dict(td)
        x = d.get(self.src_key, None)
        if not isinstance(x, Tensor):
            return as_tensordict(d)
        noise = torch.randn_like(x) * self.sigma
        d[self.out_key] = x + noise
        return as_tensordict(d)


@dataclass(frozen=True, slots=True)
class GraphBatch:
    """No-op placeholder (graph batching will be implemented in Phase 2)."""

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        return td


@dataclass(frozen=True, slots=True)
class Compose:
    transforms: list[Transform]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        out = td
        for t in self.transforms:
            out = t(out)
        return out


def build_transform(cfg: TransformConfig) -> Transform:
    if isinstance(cfg, RenameKeysTransformConfig):
        return RenameKeys(mapping=cfg.mapping)
    if isinstance(cfg, CastDtypeTransformConfig):
        return CastDtype(dtypes={k: _dtype_from_str(v) for k, v in cfg.dtypes.items()})
    if isinstance(cfg, AddMaskTransformConfig):
        return AddMask(
            src_key=cfg.src_key,
            mask_key=cfg.mask_key,
            ignore_index=cfg.ignore_index,
        )
    if isinstance(cfg, TokenShiftTransformConfig):
        return TokenShift(
            src_key=cfg.src_key,
            input_key=cfg.input_key,
            target_key=cfg.target_key,
        )
    if isinstance(cfg, GaussianNoiseTransformConfig):
        return GaussianNoise(
            src_key=cfg.src_key,
            out_key=cfg.out_key,
            sigma=cfg.sigma,
        )
    if isinstance(cfg, GraphBatchTransformConfig):
        return GraphBatch()
    raise TypeError(f"Unknown TransformConfig {type(cfg).__name__}")


def build_pipeline(payload: Any) -> Compose:
    """Build a transform pipeline from a manifest payload."""
    if payload is None:
        return Compose(transforms=[])
    if isinstance(payload, Compose):
        return payload
    # Allow either: {transforms: [...]} or a bare list.
    if isinstance(payload, dict) and "transforms" in payload:
        payload = payload["transforms"]
    if not isinstance(payload, list):
        raise TypeError("transforms must be a list or {transforms: [...]}")

    from pydantic import TypeAdapter

    ta = TypeAdapter(list[TransformConfig])
    cfgs = ta.validate_python(payload)
    return Compose(transforms=[build_transform(cfg) for cfg in cfgs])

