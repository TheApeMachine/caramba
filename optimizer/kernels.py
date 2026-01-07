"""Hardware abstraction layer (HAL) for high-impact kernels.

Caramba operates on a strict kernel policy:
- Dispatch deterministically to the best supported kernel path.
- Validate required kernel backends at startup (see `optimizer/kernel_registry.py`).
- If an expected fast path is unavailable, raise immediately (no silent fallbacks).

Notes:
- Many CUDA code paths currently rely on PyTorch's native CUDA kernels for norms/RoPE.
  This module keeps the public API stable while dispatching to the best validated backend.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from torch._dynamo import disable as _dynamo_disable

from caramba.optimizer.kernel_registry import KERNELS


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@_dynamo_disable
def rmsnorm(*, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""
    if x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="RMSNorm on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype == torch.float16,
            msg=f"RMSNorm on MPS requires fp16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.metal import rmsnorm_fp16

        return rmsnorm_fp16(x=x, weight=weight, eps=float(eps))

    if x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="RMSNorm on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"RMSNorm on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.rmsnorm_triton import rmsnorm_triton

        return rmsnorm_triton(x=x, weight=weight, eps=float(eps))

    x_f = x.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y = (x_f * inv_rms).to(dtype=x.dtype)
    if weight is not None:
        y = y * weight
    return y


@_dynamo_disable
def rope_apply(*, x: Tensor, cos: Tensor, sin: Tensor, rot_dim: int) -> Tensor:
    """Apply RoPE using cos/sin tables for the sequence window.

    Expects:
    - x: (B, H, T, D)
    - cos/sin: (T, rot_dim/2)
    """
    if x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="RoPE on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype == torch.float16,
            msg=f"RoPE on MPS requires fp16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.metal import rope_fp16

        return rope_fp16(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))

    if x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="RoPE on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"RoPE on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.rope_triton import rope_triton

        return rope_triton(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))

    T = int(x.shape[2])
    cos2 = cos[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin2 = sin[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    rot = int(rot_dim)
    x_rot = x[..., :rot]
    x_pass = x[..., rot:]
    # HF Llama applies rotate_half on a half-split representation:
    # y1 = x1*cos - x2*sin
    # y2 = x1*sin + x2*cos
    x1 = x_rot[..., : rot // 2]
    x2 = x_rot[..., rot // 2 : rot]
    y1 = x1 * cos2 - x2 * sin2
    y2 = x1 * sin2 + x2 * cos2
    return torch.cat([y1, y2, x_pass], dim=-1)


@_dynamo_disable
def layernorm(*, x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    """LayerNorm over the last dimension.

    This matches PyTorch's `F.layer_norm(x, normalized_shape=(D,))` behavior.
    """
    if x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="LayerNorm on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype == torch.float16,
            msg=f"LayerNorm on MPS requires fp16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.metal import layernorm_fp16

        return layernorm_fp16(x=x, weight=weight, bias=bias, eps=float(eps))

    if x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="LayerNorm on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"LayerNorm on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.layernorm_triton import layernorm_triton

        return layernorm_triton(x=x, weight=weight, bias=bias, eps=float(eps))

    D = int(x.shape[-1])
    return F.layer_norm(x, normalized_shape=(D,), weight=weight, bias=bias, eps=float(eps))


@_dynamo_disable
def attention_decode(
    *,
    q_sem: Tensor,
    q_geo: Tensor,
    k_sem: Tensor,
    k_geo: Tensor,
    v: Tensor,
    k_sem_null: Tensor | None = None,
    k_geo_null: Tensor | None = None,
    v_null: Tensor | None = None,
    sem_scale: float | None = None,
    geo_scale: float | None = None,
) -> Tensor:
    """Fused decode attention (HAL).

    Current supported fast paths:
    - MPS (Metal): decoupled DBA decode (fp16)

    Signature (kwargs-only):
      q_sem, q_geo, k_sem, k_geo, v,
      k_sem_null=None, k_geo_null=None, v_null=None,
      sem_scale=None, geo_scale=None
    """
    if q_sem.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="Attention decode on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            q_sem.dtype == torch.float16,
            msg=f"Attention decode on MPS requires fp16, got dtype={q_sem.dtype}.",
        )
        from caramba.optimizer.metal import dba_decode_fp16

        return dba_decode_fp16(
            q_sem=q_sem,
            q_geo=q_geo,
            k_sem=k_sem,
            k_geo=k_geo,
            v=v,
            k_sem_null=k_sem_null,
            k_geo_null=k_geo_null,
            v_null=v_null,
            sem_scale=sem_scale,
            geo_scale=geo_scale,
        )

    raise RuntimeError(
        "attention_decode: no supported backend for this device/dtype.\n"
        f"device={q_sem.device.type} dtype={q_sem.dtype}\n"
        "Use the decoupled attention fused decode paths (CUDA Triton) or Metal DBA decode (MPS fp16)."
    )


@_dynamo_disable
def scan(
    *,
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
) -> Tensor:
    """Fused scan/SSM kernels (HAL).

    Signature (kwargs-only):
      x, dt, A, B, C, D
    """
    if x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="SSM scan on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype == torch.float16,
            msg=f"SSM scan on MPS requires fp16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.metal import MetalSSMSelectiveScan

        return MetalSSMSelectiveScan().run(x=x, dt=dt, A=A, B=B, C=C, D=D)

    if x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="SSM scan on CUDA requires Triton kernels to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"SSM scan on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        from caramba.optimizer.fused_ssm import fused_selective_scan

        return fused_selective_scan(x, dt, A, B, C, D)

    raise RuntimeError(
        "scan: no supported backend for this device/dtype.\n"
        f"device={x.device.type} dtype={x.dtype}\n"
        "Supported backends: Metal (MPS fp16), Triton (CUDA fp16/bf16)."
    )


@_dynamo_disable
def adamw_step(
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
    """Fused AdamW update (HAL).

    This is the low-level per-tensor update used by `AdamWMaster` when fused.
    """
    if p.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="AdamW step on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            p.dtype == torch.float16,
            msg=f"AdamW step on MPS requires fp16 params, got dtype={p.dtype}.",
        )
        from caramba.optimizer.metal import AdamWMasterStep

        AdamWMasterStep().run(
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
            verbose_build=False,
        )
        return

    if p.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="AdamW step on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            p.dtype in (torch.float16, torch.bfloat16),
            msg=f"AdamW step on CUDA requires fp16/bf16 params, got dtype={p.dtype}.",
        )
        _require(
            grad.dtype == p.dtype,
            msg="AdamW step on CUDA requires grad dtype to match param dtype.",
        )
        _require(
            master.dtype == torch.float32 and exp_avg.dtype == torch.float32 and exp_avg_sq.dtype == torch.float32,
            msg="AdamW step on CUDA requires fp32 master/exp_avg/exp_avg_sq.",
        )
        _require(
            p.is_contiguous() and grad.is_contiguous() and master.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous(),
            msg="AdamW step on CUDA requires all tensors to be contiguous.",
        )
        from caramba.optimizer.adamw_triton import adamw_triton_master_step

        adamw_triton_master_step(
            p=p.view(-1),
            grad=grad.view(-1),
            master=master.view(-1),
            exp_avg=exp_avg.view(-1),
            exp_avg_sq=exp_avg_sq.view(-1),
            step_size=float(step_size),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            lr_wd=float(lr_wd),
        )
        return

    raise RuntimeError(
        "adamw_step: no supported backend for this device/dtype.\n"
        f"device={p.device.type} dtype={p.dtype}\n"
        "Supported backends: Metal (MPS fp16) and Triton (CUDA fp16/bf16)."
    )


def lion_step(
    *,
    p: Tensor,
    grad: Tensor,
    m: Tensor,
    lr: float,
    beta1: float,
    weight_decay: float = 0.0,
) -> None:
    """Fused Lion update (HAL)."""
    if p.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="Lion step on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            p.dtype == torch.float16,
            msg=f"Lion step on MPS requires fp16 params, got dtype={p.dtype}.",
        )
        from caramba.optimizer.metal import lion_fp16

        lion_fp16(
            p=p,
            grad=grad,
            m=m,
            lr=float(lr),
            beta1=float(beta1),
            weight_decay=float(weight_decay),
            verbose_build=False,
        )
        return

    raise RuntimeError(
        "lion_step: no supported backend for this device/dtype.\n"
        f"device={p.device.type} dtype={p.dtype}\n"
        "CUDA fused optimizer parity kernel is not available in this build."
    )
