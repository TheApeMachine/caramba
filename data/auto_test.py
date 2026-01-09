from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from caramba.data.auto import AutoDataset
from caramba.data.config import DatasetConfig, DatasetType


def test_build_token_dataset_npy_file(tmp_path: Path) -> None:
    p = tmp_path / "tiny.npy"
    np.save(str(p), np.arange(1, 11, dtype=np.int32))

    ds = AutoDataset(
        DatasetConfig(type=DatasetType.NPY, source=str(p), tokens=4)
    )

    b = ds[0]

    assert torch.equal(b["input_ids"], torch.tensor([1, 2, 3, 4], dtype=torch.long))
    assert torch.equal(b["target_ids"], torch.tensor([2, 3, 4, 5], dtype=torch.long))
