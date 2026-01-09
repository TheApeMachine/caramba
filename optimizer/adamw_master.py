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
from typing import Protocol, cast

import torch
from torch import Tensor

from caramba.optimizer.kernels import adamw_step as _adamw_step


class _AdamWStepFn(Protocol):
    def __call__(
        self,
        *,
        p: Tensor,
        grad: Tensor,
        master: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        step_size: float,
        beta1: float,
        beta2: float,
        eps: float,
        lr_wd: float,
    ) -> None: ...

adamw_step: _AdamWStepFn = cast(_AdamWStepFn, _adamw_step)


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
        fused: bool = False,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
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
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            fused = bool(group.get("fused", True))

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

                # Fast path: fused HAL kernel (Metal on MPS, Triton on CUDA).
                use_fused = False
                if fused and isinstance(g, Tensor):
                    if p.device.type == "mps":
                        use_fused = (
                            p.dtype == torch.float16
                            and g.device.type == "mps"
                            and g.dtype == torch.float16
                            and master.device.type == "mps"
                            and master.dtype == torch.float32
                            and exp_avg.device.type == "mps"
                            and exp_avg.dtype == torch.float32
                            and exp_avg_sq.device.type == "mps"
                            and exp_avg_sq.dtype == torch.float32
                        )
                    elif p.device.type == "cuda":
                        use_fused = (
                            p.dtype in (torch.float16, torch.bfloat16)
                            and g.device.type == "cuda"
                            and g.dtype == p.dtype
                            and master.device.type == "cuda"
                            and master.dtype == torch.float32
                            and exp_avg.device.type == "cuda"
                            and exp_avg.dtype == torch.float32
                            and exp_avg_sq.device.type == "cuda"
                            and exp_avg_sq.dtype == torch.float32
                        )

                if use_fused:
                    bc1 = 1.0 - beta1**step
                    bc2 = 1.0 - beta2**step
                    step_size = lr * math.sqrt(bc2) / max(1e-12, bc1)

                    adamw_step(
                        p=p,
                        grad=g,
                        master=master,
                        exp_avg=exp_avg,
                        exp_avg_sq=exp_avg_sq,
                        step_size=float(step_size),
                        beta1=float(beta1),
                        beta2=float(beta2),
                        eps=float(eps),
                        lr_wd=float(lr * wd),
                    )
                    continue

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
        return loss
