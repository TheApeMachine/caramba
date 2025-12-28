"""Objective implementations.

Objectives compute loss from (batch, outputs) without baking dataset semantics
into trainers. This is one of the key seams that makes the platform generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LossBundle:
    total: Tensor
    parts: dict[str, float]


class NextTokenCrossEntropyObjective:
    """Language modeling next-token objective (+ optional diffusion head loss)."""

    def __init__(self, *, diffusion_weight: float | None = None) -> None:
        self.diffusion_weight = diffusion_weight

    def loss(self, *, outputs: dict[str, Any], batch: dict[str, Any]) -> LossBundle:
        logits: Tensor = outputs["logits"]
        target_ids: Tensor = batch["target_ids"]

        ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.reshape(-1))
        total = ce
        parts: dict[str, float] = {"ce_loss": float(ce)}

        # Optional: diffusion head objective if the system exposes it.
        features = outputs.get("features", None)
        system = outputs.get("_system", None)
        if features is not None and system is not None and hasattr(system, "diffusion_loss"):
            try:
                diff = system.diffusion_loss(features, target_ids)  # type: ignore[attr-defined]
                w = float(self.diffusion_weight) if self.diffusion_weight is not None else float(
                    getattr(system, "diffusion_weight", 0.10)
                )
                total = total + w * diff
                parts["diff_loss"] = float(diff)
                parts["diff_weight"] = float(w)
            except Exception:
                # If diffusion isn't actually configured, keep CE-only behavior.
                pass

        return LossBundle(total=total, parts=parts)


class KeyedMSEObjective:
    """Mean squared error objective for regression-like tasks."""

    def __init__(self, *, pred_key: str = "pred", target_key: str = "targets") -> None:
        self.pred_key = str(pred_key)
        self.target_key = str(target_key)

    def loss(self, *, outputs: dict[str, Any], batch: dict[str, Any]) -> LossBundle:
        pred: Tensor = outputs[self.pred_key]
        tgt: Tensor = batch[self.target_key]
        mse = torch.mean((pred - tgt) ** 2)
        return LossBundle(total=mse, parts={"mse": float(mse)})


class KeyedCrossEntropyObjective:
    """Cross-entropy classification objective.

    Expects:
    - outputs[logits_key] shape (B, C) or (B, ..., C)
    - batch[labels_key] shape (B,) or (B, ...)
    """

    def __init__(self, *, logits_key: str = "logits", labels_key: str = "labels") -> None:
        self.logits_key = str(logits_key)
        self.labels_key = str(labels_key)

    def loss(self, *, outputs: dict[str, Any], batch: dict[str, Any]) -> LossBundle:
        logits: Tensor = outputs[self.logits_key]
        labels: Tensor = batch[self.labels_key].long()
        ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return LossBundle(total=ce, parts={"ce_loss": float(ce)})

