"""Validation helpers for the stepwise training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from caramba.runtime.tensordict_utils import TensorDictBase
from caramba.trainer.stepper.phase.loss import StudentLoss


class GlobalLossEvaluator:
    """Global loss evaluator

    This evaluator computes a small, fixed-horizon validation estimate. It is
    designed to be cheap and stable during training (few batches, no gradients).
    """

    def evaluate(
        self,
        *,
        loss: StudentLoss,
        device: torch.device,
        dist_ctx: object | None,
        loader: DataLoader[TensorDictBase],
        max_batches: int = 2,
    ) -> dict[str, float]:
        """Compute validation metrics for the current student model."""
        student = loss.student
        was_training = bool(student.training)
        student.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_diff = 0.0
        n = 0

        with torch.no_grad():
            for batch in loader:
                x = batch["input_ids"].to(device=device)
                y = batch["target_ids"].to(device=device)
                l, ce, diff = loss.compute(x=x, y=y)
                total_loss += float(l)
                total_ce += float(ce)
                if diff is not None:
                    total_diff += float(diff)
                n += 1
                if n >= int(max_batches):
                    break

        if dist_ctx is not None and n > 0:
            t = torch.tensor([total_loss, total_ce, total_diff, float(n)], device=device)
            t = dist_ctx.all_reduce(t, op="sum")  # type: ignore[attr-defined]
            total_loss = float(t[0].item())
            total_ce = float(t[1].item())
            total_diff = float(t[2].item())
            n = int(t[3].item())

        if was_training:
            student.train()

        denom = float(n) if n > 0 else 1.0
        metrics: dict[str, float] = {
            "val_loss": total_loss / denom,
            "val_ce_loss": total_ce / denom,
        }
        if bool(loss.has_diffusion):
            metrics["val_diff_loss"] = total_diff / denom
        return metrics

