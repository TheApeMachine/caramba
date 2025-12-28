"""Generic TensorDict dataset from tensor files.

This component generalizes data loading beyond (x, y) tuples by mapping named
keys to on-disk tensors.

Supported formats:
- `.npy` (NumPy), with optional mmap
- `.safetensors` (PyTorch), best-effort mmap via safetensors' safe_open
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from runtime.tensordict_utils import TensorDictBase, as_tensordict
from data.transforms import Compose, build_pipeline


def _is_npy(path: Path) -> bool:
    return path.suffix.lower() == ".npy"


def _is_safetensors(path: Path) -> bool:
    return path.suffix.lower() == ".safetensors"


class _TensorSource:
    def __len__(self) -> int: ...
    def get(self, idx: int) -> Tensor: ...


class _NpySource(_TensorSource):
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def get(self, idx: int) -> Tensor:
        x = self.arr[idx]
        return torch.from_numpy(np.asarray(x))


class _SafeTensorsSource(_TensorSource):
    def __init__(self, *, path: Path, tensor_name: str) -> None:
        self.path = path
        self.tensor_name = tensor_name

    def __len__(self) -> int:
        from safetensors import safe_open

        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            t = f.get_tensor(self.tensor_name)
            return int(t.shape[0])

    def get(self, idx: int) -> Tensor:
        from safetensors import safe_open

        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            t = f.get_tensor(self.tensor_name)
            return t[idx]


class _TensorFilesDataset(Dataset[TensorDictBase]):
    def __init__(self, *, sources: dict[str, _TensorSource], transforms: Compose) -> None:
        self.sources = sources
        self.transforms = transforms
        lengths = [len(src) for src in sources.values()]
        if not lengths:
            raise ValueError("dataset.tensors requires at least one file mapping")
        if len(set(lengths)) != 1:
            raise ValueError(f"All tensor sources must share the same length, got {sorted(set(lengths))}")
        self._len = int(lengths[0])

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, idx: int) -> TensorDictBase:
        payload: dict[str, Any] = {k: src.get(idx) for k, src in self.sources.items()}
        td = as_tensordict(payload)
        return self.transforms(td)


@dataclass(frozen=True, slots=True)
class TensorFilesDataset:
    """Manifest-referenced dataset mapping keys to tensor files.

    Config:
    - files: { key: path } where path is .npy or .safetensors
    - mmap: whether to mmap .npy files (default True)
    - safetensors_tensors: optional mapping key->tensor_name inside the file
    - transforms: optional transform pipeline payload (list or {transforms: [...]})
    """

    files: dict[str, str]
    mmap: bool = True
    safetensors_tensors: dict[str, str] | None = None
    transforms: Any | None = None

    def build(self) -> Dataset[TensorDictBase]:
        sources: dict[str, _TensorSource] = {}
        st_map = dict(self.safetensors_tensors or {})

        for key, p in dict(self.files).items():
            path = Path(str(p))
            if _is_npy(path):
                arr = np.load(path, mmap_mode="r" if bool(self.mmap) else None)
                if not isinstance(arr, np.ndarray):
                    raise TypeError(f"Expected np.ndarray for {path}, got {type(arr).__name__}")
                if arr.ndim < 1:
                    raise ValueError(f"{path} must have ndim >= 1 (leading sample dim)")
                sources[str(key)] = _NpySource(arr)
                continue

            if _is_safetensors(path):
                name = st_map.get(str(key), "")
                if not name:
                    # If unspecified, try to infer when the file has exactly one tensor.
                    from safetensors import safe_open

                    with safe_open(str(path), framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                        if len(keys) != 1:
                            raise ValueError(
                                f"{path}: safetensors file has {len(keys)} tensors; "
                                "specify safetensors_tensors.{key}=<tensor_name>"
                            )
                        name = str(keys[0])
                sources[str(key)] = _SafeTensorsSource(path=path, tensor_name=str(name))
                continue

            raise ValueError(f"Unsupported file type for key {key!r}: {path}")

        pipeline = build_pipeline(self.transforms)
        return _TensorFilesDataset(sources=sources, transforms=pipeline)

