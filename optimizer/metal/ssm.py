from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from optimizer.runtime import metal_supported

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


class _AutogradCtx(Protocol):
    saved_tensors: tuple["Tensor", ...]

    def save_for_backward(self, *tensors: "Tensor") -> None: ...


@dataclass(frozen=True, slots=True)
class MetalSSMScanAvailability:
    def available(self) -> bool:
        return metal_supported()


class MetalSSMSelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: _AutogradCtx,
        x: "Tensor",
        dt: "Tensor",
        A: "Tensor",
        Bv: "Tensor",
        Cv: "Tensor",
        D: "Tensor",
        verbose_build: bool = False,
    ) -> "Tensor":
        if x.device.type != "mps":
            raise RuntimeError("Metal SSM selective scan requires device.type == 'mps'")
        if x.dtype != torch.float16:
            raise RuntimeError("Metal SSM selective scan currently supports fp16 only")

        # Enforce contiguity/layout as required by the kernel.
        x2 = x.contiguous()
        dt2 = dt.contiguous()
        A2 = A.contiguous()
        B2 = Bv.contiguous()
        C2 = Cv.contiguous()
        D2 = D.contiguous()

        ops = load_caramba_metal_ops(verbose=bool(verbose_build))
        y, h_hist = ops.ssm_scan_forward(x2, dt2, A2, B2, C2, D2)

        # Save for backward.
        ctx.save_for_backward(x2, dt2, A2, B2, C2, D2, h_hist)
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: _AutogradCtx,
        grad_y: "Tensor",
    ) -> tuple["Tensor | None", ...]:
        (x, dt, A, Bv, Cv, D, h_hist) = ctx.saved_tensors
        if grad_y is None:
            raise RuntimeError("Metal SSM selective scan backward requires grad_y")
        if grad_y.device.type != "mps":
            raise RuntimeError("Metal SSM selective scan backward requires MPS grad_y")
        if grad_y.dtype != torch.float16:
            grad_y = grad_y.to(dtype=torch.float16)
        grad_y2 = grad_y.contiguous()

        ops = load_caramba_metal_ops(verbose=False)
        grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D = ops.ssm_scan_backward(
            grad_y2, x, dt, A, Bv, Cv, D, h_hist
        )
        # No grad for verbose_build kwarg.
        return (grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D, None)


@dataclass(frozen=True, slots=True)
class MetalSSMSelectiveScan:
    availability: MetalSSMScanAvailability = MetalSSMScanAvailability()

    def run(
        self,
        *,
        x: "Tensor",
        dt: "Tensor",
        A: "Tensor",
        B: "Tensor",
        C: "Tensor",
        D: "Tensor",
        verbose_build: bool = False,
    ) -> "Tensor":
        if not self.availability.available():
            raise RuntimeError("Metal SSM selective scan is not available on this runtime")
        y = MetalSSMSelectiveScanFn.apply(x, dt, A, B, C, D, bool(verbose_build))
        if not isinstance(y, torch.Tensor):
            raise TypeError("Metal SSM selective scan returned a non-tensor output")
        return y

