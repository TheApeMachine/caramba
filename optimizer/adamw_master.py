"""AdamW with fp32 master weights for fp16 models (MPS-safe).

Why this exists:
- On MPS there is no GradScaler.
- Updating fp16 weights directly with AdamW-style optimizers can overflow and
  poison weights (NaNs/Infs) after a single step.
- The standard fix is to keep fp32 "master" weights for the optimizer update,
  then copy the updated weights back to the fp16 model parameters.

This optimizer keeps the *model* parameters in their original dtype (often fp16),
but maintains AdamW state + updates in fp32.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


class AdamWMaster(torch.optim.Optimizer):
    """AdamW optimizer with fp32 master weights stored in state."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        if closure is not None:
            with torch.enable_grad():
                _ = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not isinstance(g, Tensor):
                    continue
                if g.is_sparse:
                    raise RuntimeError("AdamWMaster does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    # fp32 master copy + fp32 moments
                    m0 = p.detach()
                    state["master"] = m0.float().clone()
                    state["exp_avg"] = torch.zeros_like(state["master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master"])

                state["step"] += 1
                step = int(state["step"])

                master: Tensor = state["master"]
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                grad = g.detach().float()

                # Decoupled weight decay (AdamW)
                if wd != 0.0:
                    master.mul_(1.0 - lr * wd)

                # Moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1**step
                bc2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bc2) / max(1e-12, bc1)

                denom = exp_avg_sq.sqrt().add_(eps)
                master.addcdiv_(exp_avg, denom, value=-step_size)

                # Copy back to model dtype
                p.copy_(master.to(dtype=p.dtype))
        return None

