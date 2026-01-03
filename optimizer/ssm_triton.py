"""Triton SSM selective scan (CUDA).

Blockwise forward+backward; kernels live in `optimizer/ssm_triton_kernels_{fwd,bwd}.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from caramba.optimizer.ssm_triton_kernels_bwd import ssm_scan_block_bwd
from caramba.optimizer.ssm_triton_kernels_fwd import ssm_scan_block_fwd


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class _SSMScanMeta:
    B: int
    T: int
    D_inner: int
    D_state: int
    block_t: int
    block_d: int

    @property
    def num_t_blocks(self) -> int:
        return _cdiv(self.T, self.block_t)


class _TritonSSMSelectiveScan:
    """Fused selective scan (CUDA) using Triton block kernels.

    Intended condition:
    - CUDA tensors
    - fp16 or bf16
    - D_state <= 32
    """

    def __init__(self, *, block_t: int = 128, block_d: int = 64, num_warps: int = 4) -> None:
        self.block_t = int(block_t)
        self.block_d = int(block_d)
        self.num_warps = int(num_warps)

    def _validate(self, *, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> _SSMScanMeta:
        _require(x.device.type == "cuda", msg="Triton selective scan requires CUDA tensors.")
        _require(x.ndim == 3, msg=f"expected x as (B,T,D_inner), got {tuple(x.shape)}")
        _require(dt.shape == x.shape, msg="dt must match x shape")
        _require(A.ndim == 2, msg="A must be (D_inner,D_state)")
        _require(B.ndim == 3 and C.ndim == 3, msg="B,C must be (B,T,D_state)")
        _require(D.ndim == 1, msg="D must be (D_inner,)")

        Bsz, T, D_inner = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
        _require(int(A.shape[0]) == D_inner, msg="A.shape[0] must match D_inner.")
        D_state = int(A.shape[1])
        _require(1 <= D_state <= 32, msg=f"D_state must be in [1,32] for this kernel (got {D_state}).")
        _require(B.shape == (Bsz, T, D_state), msg="B shape mismatch")
        _require(C.shape == (Bsz, T, D_state), msg="C shape mismatch")
        _require(int(D.numel()) == D_inner, msg="D shape mismatch")

        _require(x.dtype in (torch.float16, torch.bfloat16), msg="x must be fp16 or bf16")
        _require(dt.dtype == x.dtype and A.dtype == x.dtype and B.dtype == x.dtype and C.dtype == x.dtype and D.dtype == x.dtype, msg="all inputs must share dtype")

        _require(x.stride(-1) == 1, msg="x last dim must be contiguous (stride==1)")
        _require(dt.stride(-1) == 1, msg="dt last dim must be contiguous (stride==1)")
        _require(A.stride(-1) == 1, msg="A last dim must be contiguous (stride==1)")
        _require(B.stride(-1) == 1, msg="B last dim must be contiguous (stride==1)")
        _require(C.stride(-1) == 1, msg="C last dim must be contiguous (stride==1)")
        _require(D.stride(-1) == 1, msg="D must be contiguous (stride==1)")

        return _SSMScanMeta(
            B=Bsz,
            T=T,
            D_inner=D_inner,
            D_state=D_state,
            block_t=self.block_t,
            block_d=self.block_d,
        )

    def forward(self, *, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> tuple[Tensor, Tensor, _SSMScanMeta]:
        meta = self._validate(x=x, dt=dt, A=A, B=B, C=C, D=D)
        _require(ssm_scan_block_fwd is not None, msg="Triton SSM scan forward kernel is unavailable.")
        kernel_fwd: Any = ssm_scan_block_fwd

        y = torch.empty((meta.B, meta.T, meta.D_inner), device=x.device, dtype=x.dtype)
        h_states = torch.empty(
            (meta.num_t_blocks + 1, meta.B, meta.D_inner, meta.D_state),
            device=x.device,
            dtype=torch.float32,
        )
        h_states[0].zero_()

        use_bf16 = bool(x.dtype == torch.bfloat16)
        grid = (meta.B * _cdiv(meta.D_inner, meta.block_d),)
        for blk in range(meta.num_t_blocks):
            t_start = blk * meta.block_t
            kernel_fwd[grid](
                x,
                dt,
                A,
                B,
                C,
                D,
                y,
                h_states[blk],
                h_states[blk + 1],
                t_start,
                meta.T,
                meta.D_inner,
                stride_x_b=x.stride(0),
                stride_x_t=x.stride(1),
                stride_x_d=x.stride(2),
                stride_dt_b=dt.stride(0),
                stride_dt_t=dt.stride(1),
                stride_dt_d=dt.stride(2),
                stride_A_d=A.stride(0),
                stride_A_s=A.stride(1),
                stride_B_b=B.stride(0),
                stride_B_t=B.stride(1),
                stride_B_s=B.stride(2),
                stride_C_b=C.stride(0),
                stride_C_t=C.stride(1),
                stride_C_s=C.stride(2),
                stride_D=D.stride(0),
                stride_y_b=y.stride(0),
                stride_y_t=y.stride(1),
                stride_y_d=y.stride(2),
                stride_h_b=h_states.stride(1),
                stride_h_d=h_states.stride(2),
                stride_h_s=h_states.stride(3),
                D_STATE=meta.D_state,
                BLOCK_T=meta.block_t,
                BLOCK_D=meta.block_d,
                USE_BF16=use_bf16,
                num_warps=self.num_warps,
            )
        return y, h_states, meta

    def backward(
        self,
        *,
        grad_y: Tensor,
        x: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        h_states: Tensor,
        meta: _SSMScanMeta,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        _require(ssm_scan_block_bwd is not None, msg="Triton SSM scan backward kernel is unavailable.")
        _require(grad_y.shape == x.shape, msg="grad_y shape mismatch")
        kernel_bwd: Any = ssm_scan_block_bwd

        grad_x = torch.empty_like(x)
        grad_dt = torch.empty_like(dt)
        grad_A = torch.zeros((meta.D_inner, meta.D_state), device=x.device, dtype=torch.float32)
        grad_B = torch.zeros((meta.B, meta.T, meta.D_state), device=x.device, dtype=torch.float32)
        grad_C = torch.zeros((meta.B, meta.T, meta.D_state), device=x.device, dtype=torch.float32)
        grad_D = torch.zeros((meta.D_inner,), device=x.device, dtype=torch.float32)
        ag = torch.zeros((meta.num_t_blocks + 1, meta.B, meta.D_inner, meta.D_state), device=x.device, dtype=torch.float32)

        use_bf16 = bool(x.dtype == torch.bfloat16)
        grid = (meta.B * _cdiv(meta.D_inner, meta.block_d),)
        for blk in range(meta.num_t_blocks - 1, -1, -1):
            t_start = blk * meta.block_t
            kernel_bwd[grid](
                x,
                dt,
                A,
                B,
                C,
                D,
                grad_y,
                h_states[blk + 1],
                ag[blk + 1],
                grad_x,
                grad_dt,
                grad_A,
                grad_B,
                grad_C,
                grad_D,
                ag[blk],
                t_start,
                meta.T,
                meta.D_inner,
                stride_x_b=x.stride(0),
                stride_x_t=x.stride(1),
                stride_x_d=x.stride(2),
                stride_dt_b=dt.stride(0),
                stride_dt_t=dt.stride(1),
                stride_dt_d=dt.stride(2),
                stride_A_d=A.stride(0),
                stride_A_s=A.stride(1),
                stride_B_b=B.stride(0),
                stride_B_t=B.stride(1),
                stride_B_s=B.stride(2),
                stride_C_b=C.stride(0),
                stride_C_t=C.stride(1),
                stride_C_s=C.stride(2),
                stride_D=D.stride(0),
                stride_gy_b=grad_y.stride(0),
                stride_gy_t=grad_y.stride(1),
                stride_gy_d=grad_y.stride(2),
                stride_gx_b=grad_x.stride(0),
                stride_gx_t=grad_x.stride(1),
                stride_gx_d=grad_x.stride(2),
                stride_gdt_b=grad_dt.stride(0),
                stride_gdt_t=grad_dt.stride(1),
                stride_gdt_d=grad_dt.stride(2),
                stride_h_b=h_states.stride(1),
                stride_h_d=h_states.stride(2),
                stride_h_s=h_states.stride(3),
                stride_gA_d=grad_A.stride(0),
                stride_gA_s=grad_A.stride(1),
                stride_gB_b=grad_B.stride(0),
                stride_gB_t=grad_B.stride(1),
                stride_gB_s=grad_B.stride(2),
                stride_gC_b=grad_C.stride(0),
                stride_gC_t=grad_C.stride(1),
                stride_gC_s=grad_C.stride(2),
                stride_gD=grad_D.stride(0),
                stride_ag_b=ag.stride(1),
                stride_ag_d=ag.stride(2),
                stride_ag_s=ag.stride(3),
                D_STATE=meta.D_state,
                BLOCK_T=meta.block_t,
                BLOCK_D=meta.block_d,
                USE_BF16=use_bf16,
                num_warps=self.num_warps,
            )

        return (
            grad_x,
            grad_dt,
            grad_A.to(dtype=A.dtype),
            grad_B.to(dtype=B.dtype),
            grad_C.to(dtype=C.dtype),
            grad_D.to(dtype=D.dtype),
        )


class _TritonSelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
    ) -> Tensor:
        impl = _TritonSSMSelectiveScan()
        y, h_states, meta = impl.forward(x=x, dt=dt, A=A, B=B, C=C, D=D)
        ctx.impl = impl  # type: ignore[attr-defined]
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(x, dt, A, B, C, D, h_states)
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y: Tensor,
    ) -> tuple[Tensor | None, ...]:
        x, dt, A, B, C, D, h_states = ctx.saved_tensors
        impl: _TritonSSMSelectiveScan = getattr(ctx, "impl")
        meta: _SSMScanMeta = getattr(ctx, "meta")
        gx, gdt, gA, gB, gC, gD = impl.backward(
            grad_y=grad_y,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            h_states=h_states,
            meta=meta,
        )
        return (gx, gdt, gA, gB, gC, gD)


def selective_scan_triton(x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """Fused selective scan on CUDA via Triton."""
    y = _TritonSelectiveScanFn.apply(x, dt, A, B, C, D)
    if not isinstance(y, torch.Tensor):
        raise TypeError("selective_scan_triton returned a non-tensor output")
    return y
