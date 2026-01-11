from __future__ import annotations

"""HuggingFace image+label dataset component.

This is a small adapter so image classification style experiments can be
manifest-driven without hard-coding MNIST or torchvision.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict


class _HFImageClsDataset(Dataset[TensorDictBase]):
    def __init__(
        self,
        *,
        ds: Any,
        image_key: str,
        label_key: str,
        grayscale: bool,
        resize: tuple[int, int] | None,
        input_key: str,
        target_key: str,
        normalize_01: bool,
    ) -> None:
        self.ds = ds
        self.image_key = str(image_key)
        self.label_key = str(label_key)
        self.grayscale = bool(grayscale)
        self.resize = resize
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.normalize_01 = bool(normalize_01)

    def __len__(self) -> int:
        return int(len(self.ds))

    def _to_tensor_chw(self, img: object) -> torch.Tensor:
        # HF "image" column items are commonly PIL Images.
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Pillow is required for dataset.hf_image_classification (pip install pillow)."
            ) from e

        if isinstance(img, Image.Image):
            im = img
        else:
            # Some datasets return dicts like {"bytes":..., "path":...} which PIL can open.
            try:
                im = Image.open(img)  # type: ignore[arg-type]
            except Exception as e:
                raise TypeError(f"Unsupported image type {type(img).__name__}") from e

        if self.resize is not None:
            im = im.resize((int(self.resize[1]), int(self.resize[0])))

        if self.grayscale:
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.float32)[None, :, :]  # 1HW
        else:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)  # CHW

        if self.normalize_01:
            arr = arr / 255.0
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def __getitem__(self, idx: int) -> TensorDictBase:
        ex = self.ds[int(idx)]
        if not isinstance(ex, dict):
            ex = dict(ex)
        img = ex.get(self.image_key, None)
        y = ex.get(self.label_key, None)
        if img is None:
            raise KeyError(f"Missing image_key={self.image_key!r} in example")
        if y is None:
            raise KeyError(f"Missing label_key={self.label_key!r} in example")
        x_t = self._to_tensor_chw(img)
        y_t = torch.tensor(int(y), dtype=torch.long)
        return as_tensordict({self.input_key: x_t, self.target_key: y_t})


@dataclass(frozen=True, slots=True)
class HFImageClassificationDataset:
    """Manifest-level dataset for (image, label) tasks via HF datasets."""

    dataset: str
    split: str = "train"
    name: str | None = None

    image_key: str = "image"
    label_key: str = "label"

    input_key: str = "inputs"
    target_key: str = "targets"

    grayscale: bool = True
    resize_h: int | None = None
    resize_w: int | None = None
    normalize_01: bool = True

    def build(self) -> Dataset[TensorDictBase]:
        try:
            import datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("Need HuggingFace datasets. Install: pip install datasets") from e

        ds = datasets.load_dataset(
            str(self.dataset),
            str(self.name) if self.name is not None else None,
            split=str(self.split),
        )

        resize: tuple[int, int] | None = None
        if self.resize_h is not None and self.resize_w is not None:
            resize = (int(self.resize_h), int(self.resize_w))

        return _HFImageClsDataset(
            ds=ds,
            image_key=str(self.image_key),
            label_key=str(self.label_key),
            grayscale=bool(self.grayscale),
            resize=resize,
            input_key=str(self.input_key),
            target_key=str(self.target_key),
            normalize_01=bool(self.normalize_01),
        )

