"""Optimizer builder for the phase-based training loop."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class PhaseOptimizerBuilder:
    """Optimizer builder

    The phase-based global fine-tuning loop needs a small set of optimizers
    that cover most research workflows (AdamW, SGD, Lion). This builder keeps
    the selection logic out of the training loop itself.
    """

    def build(self, *, train: object, params) -> Optimizer:
        """Build an optimizer from a train config object and a parameter iterable."""
        opt_name = str(getattr(train, "optimizer", "adamw")).lower()
        weight_decay = float(getattr(train, "weight_decay", 0.0))
        fused_opt = bool(getattr(train, "fused_optimizer", False))

        if opt_name in ("adamw", "adam"):
            return torch.optim.AdamW(
                params,
                lr=float(getattr(train, "lr")),
                weight_decay=float(weight_decay),
            )
        if opt_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=float(getattr(train, "lr")),
                weight_decay=float(weight_decay),
            )
        if opt_name == "lion":
            from caramba.optimizer.lion import Lion

            return Lion(
                params,
                lr=float(getattr(train, "lr")),
                weight_decay=float(weight_decay),
                fused=bool(fused_opt),
            )
        raise ValueError(f"Unknown optimizer {opt_name!r}")

