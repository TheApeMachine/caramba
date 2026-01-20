from __future__ import annotations

"""CCL system wrapper.

This is a small nn.Module so it can participate in the same orchestration path
as other Caramba systems (e.g., can be returned from a trainer and inspected).
"""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ccl.context_counts import ClassCountsModel, ContextTemplate, loglik_grids
from ccl.patch_vq import PatchKMeansVQ


def _as_numpy_images(x: object) -> np.ndarray:
    if isinstance(x, Tensor):
        t = x.detach().cpu()
        if t.dtype != torch.float32:
            t = t.float()
        return t.numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported image type {type(x).__name__}")


class CCLSystem(nn.Module):
    """A trained CCL classifier / generative model in token-grid space."""

    def __init__(
        self,
        *,
        # Codec config (required for tokenize/decode)
        k: int,
        patch: int,
        stride: int,
        # Trained artifacts
        centers: np.ndarray,
        models: Sequence[ClassCountsModel],
        label_to_class: dict[int, int],
        class_to_label: dict[int, int],
        # Optional knobs
        tokenize_batch_size: int = 256,
        input_key: str = "inputs",
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.patch = int(patch)
        self.stride = int(stride)
        self.tokenize_batch_size = int(tokenize_batch_size)
        self.input_key = str(input_key)

        self.codec = PatchKMeansVQ(k=int(k), patch=int(patch), stride=int(stride))
        self.centers = np.asarray(centers, dtype=np.float32)
        self.models = list(models)
        self.label_to_class = {int(k): int(v) for k, v in dict(label_to_class).items()}
        self.class_to_label = {int(k): int(v) for k, v in dict(class_to_label).items()}

        # Register a dummy parameter so .to(device=...) works uniformly (no-op model).
        self._dummy = nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)

    def forward(self, batch: Any) -> dict[str, Tensor]:
        """Return classification logits from code lengths.

        Expected batch:
          - dict-like with key "inputs" (configurable upstream) containing images
        Output:
          - {"logits": (B,C) float32}
        """
        if not isinstance(batch, dict):
            batch = dict(batch)
        x = batch.get(str(self.input_key), None)
        if x is None:
            raise KeyError(f"CCLSystem.forward expects batch[{self.input_key!r}]")
        images = _as_numpy_images(x)
        tokens = self.codec.tokenize(images, centers=self.centers, batch_size=int(self.tokenize_batch_size))
        # logits = log p(x|class)
        logits = np.stack([loglik_grids(m, tokens) for m in self.models], axis=1).astype(
            np.float32, copy=False
        )
        return {"logits": torch.as_tensor(logits, device=self._dummy.device)}

    @torch.no_grad()
    def predict(self, images: object) -> Tensor:
        """Predict class indices (contiguous 0..C-1) for a batch of images."""
        arr = _as_numpy_images(images)
        tokens = self.codec.tokenize(arr, centers=self.centers, batch_size=int(self.tokenize_batch_size))
        scores = np.stack([loglik_grids(m, tokens) for m in self.models], axis=1)
        out = np.asarray(np.argmax(scores, axis=1), dtype=np.int64)
        return torch.as_tensor(out, device=self._dummy.device)

