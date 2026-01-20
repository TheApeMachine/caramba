from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from optimizer.runtime import metal_supported

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


def metal_adamw_available() -> bool:
    return metal_supported()


class AdamWMasterStep:
    def run(
        self,
        *,
        p: "Tensor",
        grad: "Tensor",
        master: "Tensor",
        exp_avg: "Tensor",
        exp_avg_sq: "Tensor",
        step_size: float,
        beta1: float,
        beta2: float,
        eps: float,
        lr_wd: float,
        verbose_build: bool = False,
    ) -> None:
        if p.device.type != "mps":
            raise RuntimeError("Metal AdamWMaster requires device.type == 'mps'")
        if p.dtype not in (torch.float16, torch.float32) or grad.dtype != p.dtype:
            raise RuntimeError("Metal AdamWMaster requires fp16/fp32 p/grad (matching)")
        if master.dtype != torch.float32 or exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
            raise RuntimeError("Metal AdamWMaster requires fp32 master/exp_avg/exp_avg_sq")
        if p.shape != grad.shape:
            raise RuntimeError("Metal AdamWMaster requires p and grad shapes to match")
        if master.shape != p.shape or exp_avg.shape != p.shape or exp_avg_sq.shape != p.shape:
            raise RuntimeError("Metal AdamWMaster requires state tensors to match param shape")

        ops = load_caramba_metal_ops(verbose=bool(verbose_build))
        # Extension expects flat contiguous buffers.
        ops.adamw_master_step(
            p.contiguous().view(-1),
            grad.contiguous().view(-1),
            master.contiguous().view(-1),
            exp_avg.contiguous().view(-1),
            exp_avg_sq.contiguous().view(-1),
            float(step_size),
            float(beta1),
            float(beta2),
            float(eps),
            float(lr_wd),
        )

