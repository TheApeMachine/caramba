from __future__ import annotations

import numpy as np
import torch
from typing import Any

from caramba.data.tensors import TensorFilesDataset


def test_dataset_tensors_npy_mmap_basic(tmp_path) -> None:
    x = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randn(10, 2).astype(np.float32)
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, x)
    np.save(y_path, y)

    ds_comp = TensorFilesDataset(files={"inputs": str(x_path), "targets": str(y_path)}, mmap=True)
    ds: Any = ds_comp.build()
    assert len(ds) == 10
    item = ds[0]
    assert "inputs" in item and "targets" in item
    assert isinstance(item["inputs"], torch.Tensor)
    assert isinstance(item["targets"], torch.Tensor)
    assert tuple(item["inputs"].shape) == (3,)
    assert tuple(item["targets"].shape) == (2,)


def test_dataset_tensors_transforms_token_shift(tmp_path) -> None:
    toks = np.arange(0, 60, dtype=np.int64).reshape(10, 6)
    p = tmp_path / "tokens.npy"
    np.save(p, toks)

    ds_comp = TensorFilesDataset(
        files={"tokens": str(p)},
        transforms=[
            {"type": "token_shift", "src_key": "tokens", "input_key": "input_ids", "target_key": "target_ids"}
        ],
    )
    ds = ds_comp.build()
    item = ds[3]
    assert tuple(item["input_ids"].shape) == (5,)
    assert tuple(item["target_ids"].shape) == (5,)
    assert torch.equal(item["target_ids"], item["input_ids"] + 1)

