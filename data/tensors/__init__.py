"""Tensor files dataset package

Loads multiple named tensors from disk files and combines them into TensorDicts,
enabling datasets beyond simple (x, y) pairs. Supports both NumPy arrays and
safetensors formats, making it easy to work with preprocessed multi-modal data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from torch.utils.data import Dataset

from data.transforms import Compose
from data.transforms.add_mask import AddMask
from data.transforms.cast_dtype import CastDtype, _dtype_from_str
from data.transforms.gaussian_noise import GaussianNoise
from data.transforms.rename_keys import RenameKeys
from data.transforms.token_shift import TokenShift
from data.tensors.dataset import TensorFilesDataset as _TensorFilesDataset
from data.tensors.npy_source import NpySource
from data.tensors.safetensors_source import SafeTensorsSource
from data.tensors.source import TensorSource
from data.tensors.utils import is_npy, is_safetensors
from runtime.tensordict_utils import TensorDictBase


@dataclass(frozen=True, slots=True)
class TensorFilesDataset:
    """Tensor files dataset component

    Manifest-level dataset that maps named keys to tensor files on disk,
    supporting both NumPy arrays and safetensors formats. This makes it easy
    to load multi-modal or multi-feature datasets from config files.
    """

    files: dict[str, str]
    mmap: bool = True
    safetensors_tensors: dict[str, str] | None = None
    transforms: Any | None = None

    def build(self) -> Dataset[TensorDictBase]:
        """Build dataset from file mappings

        Creates tensor sources for each file, automatically detecting format
        and handling safetensors name resolution. Returns a dataset ready for
        use in training loops.
        """
        sources: dict[str, TensorSource] = {}
        st_map = dict(self.safetensors_tensors or {})

        for key, p in self.files.items():
            path = Path(str(p))
            if is_npy(path):
                arr = np.load(path, mmap_mode="r" if bool(self.mmap) else None)
                if not isinstance(arr, np.ndarray):
                    raise TypeError(f"Expected np.ndarray for {path}, got {type(arr).__name__}")
                if arr.ndim < 1:
                    raise ValueError(f"{path} must have ndim >= 1 (leading sample dim)")
                sources[str(key)] = NpySource(arr)
                continue

            if is_safetensors(path):
                name = st_map.get(str(key), "")
                if not name:
                    # If unspecified, try to infer when the file has exactly one tensor.
                    with safe_open(str(path), framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                        if len(keys) != 1:
                            raise ValueError(
                                f"{path}: safetensors file has {len(keys)} tensors; "
                                "specify safetensors_tensors.{key}=<tensor_name>"
                            )
                        name = str(keys[0])
                sources[str(key)] = SafeTensorsSource(path=path, tensor_name=str(name))
                continue

            raise ValueError(f"Unsupported file type for key {key!r}: {path}")

        # Handle transforms - if None or empty, create empty Compose
        if self.transforms is None:
            pipeline = Compose(transforms=[])
        elif isinstance(self.transforms, Compose):
            pipeline = self.transforms
        else:
            # If transforms is a list, convert dict configs to Transform instances
            if isinstance(self.transforms, list):
                transform_instances = []
                for t in self.transforms:
                    if isinstance(t, dict):
                        # Convert dict config to Transform instance
                        transform_type = t.get("type")
                        if transform_type == "token_shift":
                            transform_instances.append(
                                TokenShift(
                                    src_key=str(t["src_key"]),
                                    input_key=str(t["input_key"]),
                                    target_key=str(t["target_key"]),
                                )
                            )
                        elif transform_type == "rename_keys":
                            transform_instances.append(RenameKeys(mapping=dict(t["mapping"])))
                        elif transform_type == "cast_dtype":
                            dtypes_dict = {}
                            for key, dtype_str in t.get("dtypes", {}).items():
                                dtypes_dict[str(key)] = _dtype_from_str(str(dtype_str))
                            transform_instances.append(CastDtype(dtypes=dtypes_dict))
                        elif transform_type == "add_mask":
                            transform_instances.append(
                                AddMask(
                                    src_key=str(t.get("src_key", "targets")),
                                    mask_key=str(t.get("mask_key", "mask")),
                                    ignore_index=int(t.get("ignore_index", -100)),
                                )
                            )
                        elif transform_type == "gaussian_noise":
                            transform_instances.append(
                                GaussianNoise(
                                    src_key=str(t.get("src_key", "inputs")),
                                    out_key=str(t.get("out_key", "noisy")),
                                    sigma=float(t.get("sigma", 1.0)),
                                )
                            )
                        else:
                            raise ValueError(f"Unknown transform type: {transform_type}")
                    else:
                        # Already a Transform instance
                        transform_instances.append(t)
                pipeline = Compose(transforms=transform_instances)
            else:
                raise TypeError(f"transforms must be Compose, list, or None, got {type(self.transforms).__name__}")

        return _TensorFilesDataset(sources=sources, transforms=pipeline)


__all__ = [
    "TensorFilesDataset",
    "TensorSource",
    "NpySource",
    "SafeTensorsSource",
]
