"""Objective implementations.

Objectives compute a *scalar* loss tensor from (batch, outputs) using a strict,
dictionary-based protocol:

- batch:   dict[str, Tensor]
- outputs: dict[str, Tensor]

The particular tensor keys are configured via the manifest so trainers stay
completely model/task agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from caramba.runtime.tensordict_utils import TensorDictBase

TensorDict = TensorDictBase
MetricDict = dict[str, float]


def _require_tensor(d: Mapping[str, Any], key: str, *, where: str) -> Tensor:
    if key not in d:
        raise KeyError(f"Missing key {key!r} in {where}.")
    v = d[key]
    if not isinstance(v, Tensor):
        raise TypeError(f"Expected {where}[{key!r}] to be a Tensor, got {type(v).__name__}")
    return v


class KeyedMSEObjective:
    """Mean squared error objective for regression-like tasks."""

    def __init__(self, *, pred_key: str = "pred", target_key: str = "targets") -> None:
        self.pred_key = str(pred_key)
        self.target_key = str(target_key)

    def loss(self, *, outputs: TensorDict, batch: TensorDict) -> Tensor:
        pred = _require_tensor(outputs, self.pred_key, where="outputs")
        tgt = _require_tensor(batch, self.target_key, where="batch")
        return torch.mean((pred - tgt) ** 2)

    def metrics(self, *, outputs: TensorDict, batch: TensorDict, loss: Tensor) -> MetricDict:
        _ = outputs
        _ = batch
        return {"mse": float(loss.detach())}


class KeyedCrossEntropyObjective:
    """Cross-entropy classification objective.

    Expects:
    - outputs[logits_key] shape (B, C) or (B, ..., C)
    - batch[labels_key] shape (B,) or (B, ...)
    """

    def __init__(
        self,
        *,
        logits_key: str = "logits",
        labels_key: str = "labels",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        self.logits_key = str(logits_key)
        self.labels_key = str(labels_key)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def loss(self, *, outputs: TensorDict, batch: TensorDict) -> Tensor:
        logits = _require_tensor(outputs, self.logits_key, where="outputs")
        labels = _require_tensor(batch, self.labels_key, where="batch").long()
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def metrics(self, *, outputs: TensorDict, batch: TensorDict, loss: Tensor) -> MetricDict:
        _ = outputs
        _ = batch
        return {"ce_loss": float(loss.detach())}


class NextTokenCrossEntropyObjective(KeyedCrossEntropyObjective):
    """Legacy name for a keyed cross entropy objective (LM next-token by default)."""

    def __init__(
        self,
        *,
        logits_key: str = "logits",
        target_key: str = "target_ids",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            logits_key=logits_key,
            labels_key=target_key,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        # Backwards-compatible alias.
        self.target_key = self.labels_key

