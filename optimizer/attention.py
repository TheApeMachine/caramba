"""Attention training kernels and dispatch.

Caramba's attention policy:
- CUDA training uses custom Triton FlashAttention (forward+backward).
- MPS training uses custom Metal fused attention (forward+backward).

This module provides a single composable object for attention execution so layers
don't need to embed backend-specific logic.
"""

from __future__ import annotations

import os
import contextlib
import torch
import torch.nn.functional as F
from torch import Tensor

from console import logger
import kernels


def _sdpa_pytorch(
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    scale: float,
    dropout_p: float,
) -> Tensor:
    """Run PyTorch SDPA with a preference for low-memory kernels on CUDA.

    On older CUDA GPUs (e.g. sm_5.x) the default SDPA kernel selection may fall
    back to the "math" implementation, which can materialize large (T×T) attention
    tensors and OOM for moderate seq lengths across many layers.
    """
    # PyTorch's memory-efficient SDPA kernel has dtype constraints that can bite when
    # the surrounding training stack forces bf16 autocast (e.g. accelerate config).
    # If QKV arrive as bf16 on CUDA, downcast to fp16 for the SDPA op, then cast back.
    # This both enables the mem-efficient kernel and avoids bf16 unsupported paths on pre-Ampere GPUs.
    orig_dtype = q.dtype
    if q.device.type == "cuda" and orig_dtype == torch.bfloat16:
        q = q.to(dtype=torch.float16)
        k = k.to(dtype=torch.float16)
        v = v.to(dtype=torch.float16)

    if q.device.type == "cuda":
        # Prefer new API if available; torch.backends.cuda.sdp_kernel is deprecated.
        sdpa_cm = None
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore[attr-defined]

            # Restrict SDPA to the efficient backend only.
            sdpa_cm = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION], True)  # type: ignore[misc]
        except Exception:
            sdpa_cm = None

        prefer = str(os.environ.get("CARAMBA_SDPA_PREFER", "mem_efficient")).lower()
        if prefer in ("mem_efficient", "mem-efficient", "memory_efficient", "memory-efficient"):
            try:
                # Accelerate can enable bf16 autocast globally; SDPA's efficient backend
                # often requires fp16/fp32. Disable autocast locally so our fp16 cast sticks.
                autocast_off = (
                    torch.autocast(device_type="cuda", enabled=False)
                    if hasattr(torch, "autocast")
                    else contextlib.nullcontext()
                )
                if sdpa_cm is not None:
                    with sdpa_cm, autocast_off:
                        out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=None,
                            dropout_p=float(dropout_p),
                            is_causal=bool(causal),
                            scale=float(scale),
                        )
                else:
                    # Back-compat for older PyTorch.
                    with autocast_off, torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                        enable_flash=False,
                        enable_mem_efficient=True,
                        enable_math=False,
                    ):
                        out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=None,
                            dropout_p=float(dropout_p),
                            is_causal=bool(causal),
                            scale=float(scale),
                        )
                return out.to(dtype=orig_dtype) if out.dtype != orig_dtype else out
            except Exception:
                # Fall through to default SDPA selection.
                pass

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=float(dropout_p),
        is_causal=bool(causal),
        scale=float(scale),
    )
    return out.to(dtype=orig_dtype) if out.dtype != orig_dtype else out

class AttentionTraining:
    """Full-sequence attention for training.

    This is a strict kernel contract:
    - No arbitrary attention masks (padding masks) in the fused CUDA path
    - Dropout is supported (mask is regenerated in backward from a saved seed)
    """
    def __init__(self):
        self._cuda_flash_warned = False
        if torch.cuda.is_available():
            # FlashAttention kernels typically require Ampere (sm_8.x) or newer.
            # On older GPUs (e.g. sm_5.2), we must fall back to PyTorch SDPA.
            try:
                cc_major, _cc_minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            except Exception:
                cc_major = 0
            if int(cc_major) < 8:
                self.cuda_flash_sdpa = None
                return

            # Prefer Kernel Hub FlashAttention if a compatible build exists for this environment.
            # Otherwise, fall back to PyTorch SDPA (works everywhere).
            try:
                from kernels import has_kernel

                self.cuda_flash_sdpa = (
                    kernels.get_kernel("kernels-community/flash-attn2")
                    if has_kernel("kernels-community/flash-attn2")
                    else None
                )
            except Exception as e:
                logger.warning(f"Kernel Hub flash-attn2 unavailable; falling back to PyTorch SDPA. Error: {e}")
                self.cuda_flash_sdpa = None
        elif torch.backends.mps.is_available():
            # The kernels-community Metal Flash-SDPA path has been observed to trigger
            # MTLCommandBuffer encoder state asserts on some Apple GPUs/toolchains.
            # Default to PyTorch's native SDPA on MPS unless explicitly enabled.
            use_mps_flash = str(os.environ.get("CARAMBA_USE_MPS_FLASH_SDPA", "0")).lower() in ("1", "true", "yes")
            self.metal_flash_sdpa = (
                kernels.get_kernel("kernels-community/metal-flash-sdpa") if use_mps_flash else None
            )
        else:
            raise RuntimeError("AttentionTraining: no supported device found.")

    @staticmethod
    def _cu_seqlens_fixed(*, B: int, T: int, device: torch.device) -> Tensor:
        # flash-attn varlen expects cu_seqlens of shape (B+1,) with inclusive prefix sums.
        # For fixed-length sequences, this is [0, T, 2T, ..., B*T].
        return torch.arange(0, (B + 1) * T, step=T, device=device, dtype=torch.int32)

    @staticmethod
    def _flatten_bhtd(x: Tensor) -> Tensor:
        # (B,H,T,D) -> (B*T,H,D)
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B * T, H, D)

    @staticmethod
    def _unflatten_bthd(x: Tensor, *, B: int, H: int, T: int, D: int) -> Tensor:
        # (B*T,H,D) -> (B,H,T,D)
        return x.view(B, T, H, D).permute(0, 2, 1, 3).contiguous()

    def _run_cuda(
        self, 
        *, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        causal: bool, 
        scale: float, 
        dropout_p: float, 
    ) -> Tensor:
        if self.cuda_flash_sdpa is None:
            # Portable fallback.
            return _sdpa_pytorch(q=q, k=k, v=v, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p))
        B, H, Tq, D = q.shape
        _, _, Tk, _ = k.shape
        cu_q = self._cu_seqlens_fixed(B=B, T=int(Tq), device=q.device)
        cu_k = self._cu_seqlens_fixed(B=B, T=int(Tk), device=k.device)
        try:
            out = self.cuda_flash_sdpa.flash_attn_varlen_func(
                q=self._flatten_bhtd(q),
                k=self._flatten_bhtd(k),
                v=self._flatten_bhtd(v),
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=int(Tq),
                max_seqlen_k=int(Tk),
                dropout_p=float(dropout_p),
                softmax_scale=float(scale),
                causal=bool(causal),
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
            )
        except RuntimeError as e:
            # Kernel Hub build exists, but runtime may still reject the GPU (e.g. pre-Ampere).
            if not self._cuda_flash_warned:
                self._cuda_flash_warned = True
                logger.warning(f"FlashAttention kernel failed at runtime; falling back to PyTorch SDPA. Error: {e}")
            return _sdpa_pytorch(q=q, k=k, v=v, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p))
        return self._unflatten_bthd(out, B=B, H=H, T=int(Tq), D=D)

    def _run_mps(
        self, 
        *, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        causal: bool, 
        scale: float, 
        dropout_p: float, 
        attn_mask: Tensor | None = None
    ) -> Tensor:
        # Prefer PyTorch-native SDPA on MPS by default (more stable).
        if self.metal_flash_sdpa is None:
            if attn_mask is not None:
                raise RuntimeError("AttentionTraining MPS path does not support attn_mask (must be None).")
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=float(dropout_p),
                is_causal=bool(causal),
                scale=float(scale),
            )

        B, H, Tq, D = q.shape
        _, _, Tk, _ = k.shape
        cu_q = self._cu_seqlens_fixed(B=B, T=int(Tq), device=q.device)
        cu_k = self._cu_seqlens_fixed(B=B, T=int(Tk), device=k.device)
        out = self.metal_flash_sdpa.flash_attn_varlen_func(
            q=self._flatten_bhtd(q),
            k=self._flatten_bhtd(k),
            v=self._flatten_bhtd(v),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(Tq),
            max_seqlen_k=int(Tk),
            dropout_p=float(dropout_p),
            softmax_scale=float(scale),
            causal=bool(causal),
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False
        )
        return self._unflatten_bthd(out, B=B, H=H, T=int(Tq), D=D)


    def run(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        scale: float,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> Tensor:
        """Run full-sequence attention.

        Args:
            q/k/v: (B,H,T,D)
            causal: causal masking
            scale: scaling applied to QK^T
            attn_mask: must be None for the fused path
            dropout_p: dropout probability (mask is regenerated from seed in backward)
        """
        if q.device.type == "cuda":
            return self._run_cuda(
                q=q, 
                k=k, 
                v=v, 
                causal=bool(causal), 
                scale=float(scale), 
                dropout_p=float(dropout_p), 
            )
        if q.device.type == "mps":
            return self._run_mps(
                q=q, 
                k=k, 
                v=v, 
                causal=bool(causal), 
                scale=float(scale), 
                dropout_p=float(dropout_p), 
            )

        raise RuntimeError(f"AttentionTraining: unsupported device.type={q.device.type!r}.")

