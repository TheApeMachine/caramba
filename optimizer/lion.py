"""Lion optimizer (with optional Metal fast path).

Reference: https://arxiv.org/abs/2302.06675

This implementation is intentionally minimal and primarily serves as a vehicle
for MPS/Metal fused update experiments.
"""

from __future__ import annotations

import logging
import torch
from torch import Tensor

log = logging.getLogger(__name__)


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        fused: bool = False,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(x) for x in betas),
            "weight_decay": float(weight_decay),
            "fused": bool(fused),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            wd = float(group["weight_decay"])
            fused = bool(group.get("fused", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not isinstance(p.grad, Tensor):
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m: Tensor = state["m"]

                # Metal fused path (fp16 on MPS).
                if fused and p.device.type == "mps" and p.dtype == torch.float16 and g.dtype == torch.float16:
                    from optimizer.kernels import lion_step

                    lion_step(p=p, grad=g, m=m, lr=lr, beta1=beta1, weight_decay=wd)
                    # Secondary momentum update
                    m.mul_(beta2).add_(g, alpha=1.0 - beta2)
                    continue

                # Reference update (eager PyTorch).
                # m = beta1*m + (1-beta1)*g
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(torch.sign(m), alpha=-lr)
                # second momentum (not used in update, but helps stability)
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss

