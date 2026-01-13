"""CUDA/Triton AdamW master-step wrapper.

This provides a small, validated wrapper around the Triton kernel so the HAL
(`optimizer.kernels.adamw_step`) can dispatch without importing Triton at runtime
call-sites (startup validation ensures kernels exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from caramba.kernel.triton.adamw_triton_kernels import adamw_master_step


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class _AdamWMeta:
    n: int
    block: int
    use_bf16: bool


class _AdamWTriton:
    """Fused AdamW master-step on CUDA via Triton."""

    def _validate(
        self,
        *,
        p: Tensor,
        grad: Tensor,
        master: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
    ) -> _AdamWMeta:
        _require(p.device.type == "cuda", msg="AdamW Triton requires CUDA tensors.")
        _require(p.dtype in (torch.float16, torch.bfloat16), msg="AdamW Triton supports fp16/bf16 params.")
        _require(grad.device == p.device, msg="AdamW grad must be on same device as params.")
        _require(grad.dtype == p.dtype, msg="AdamW grad dtype must match params dtype.")
        _require(master.device == p.device and exp_avg.device == p.device and exp_avg_sq.device == p.device, msg="AdamW state must be on same device as params.")
        _require(master.dtype == torch.float32, msg="AdamW master must be fp32.")
        _require(exp_avg.dtype == torch.float32 and exp_avg_sq.dtype == torch.float32, msg="AdamW moments must be fp32.")
        _require(p.is_contiguous(), msg="AdamW Triton requires contiguous params.")
        _require(grad.is_contiguous(), msg="AdamW Triton requires contiguous grads.")
        _require(master.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous(), msg="AdamW Triton requires contiguous state tensors.")
        _require(p.numel() == grad.numel() == master.numel() == exp_avg.numel() == exp_avg_sq.numel(), msg="AdamW Triton requires all tensors to have identical numel.")

        n = int(p.numel())
        _require(n > 0, msg="AdamW Triton requires n > 0.")
        block = 1024
        return _AdamWMeta(n=n, block=block, use_bf16=bool(p.dtype == torch.bfloat16))

    def run(
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
    ) -> None:
        meta = self._validate(p=p, grad=grad, master=master, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        _require(adamw_master_step is not None, msg="AdamW Triton kernel is unavailable.")
        k: Any = adamw_master_step

        grid = (_cdiv(meta.n, meta.block),)
        k[grid](
            p,
            grad,
            master,
            exp_avg,
            exp_avg_sq,
            n_elements=meta.n,
            step_size=float(step_size),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            lr_wd=float(lr_wd),
            USE_BF16=int(meta.use_bf16),
            BLOCK=meta.block,
            num_warps=4,
        )


_IMPL = _AdamWTriton()


def adamw_triton_master_step(
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
) -> None:
    """Run the fused AdamW master-step update (CUDA/Triton)."""
    _IMPL.run(
        p=p,
        grad=grad,
        master=master,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        step_size=float(step_size),
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(eps),
        lr_wd=float(lr_wd),
    )

