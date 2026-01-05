"""Attention training kernels and dispatch.

Caramba's attention policy:
- CUDA training uses custom Triton FlashAttention (forward+backward).
- MPS training uses custom Metal fused attention (forward+backward).

This module provides a single composable object for attention execution so layers
don't need to embed backend-specific logic.
"""

from __future__ import annotations

import torch
from torch import Tensor

from caramba.optimizer.kernel_registry import KERNELS


class AttentionTraining:
    """Full-sequence attention for training.

    This is a strict kernel contract:
    - No arbitrary attention masks (padding masks) in the fused CUDA path
    - Dropout is supported (mask is regenerated in backward from a saved seed)
    """

    def _require(self, cond: bool, *, msg: str) -> None:
        if not cond:
            raise RuntimeError(msg)

    def _validate_common(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
        dropout_p: float,
    ) -> None:
        self._require(q.shape == k.shape == v.shape, msg="AttentionTraining requires q/k/v shapes to match.")
        self._require(q.dtype == k.dtype == v.dtype, msg="AttentionTraining requires q/k/v dtypes to match.")
        self._require(q.ndim == 4, msg="AttentionTraining expects q/k/v shape (B,H,T,D).")
        self._require(attn_mask is None, msg="AttentionTraining fused path does not support attn_mask; use packing/no-padding.")
        self._require(0.0 <= float(dropout_p) < 1.0, msg="AttentionTraining requires 0 <= dropout_p < 1.")

    def _run_cuda(self, *, q: Tensor, k: Tensor, v: Tensor, causal: bool, scale: float, dropout_p: float) -> Tensor:
        self._require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="CUDA attention training requires Triton kernels validated at startup.",
        )
        from caramba.optimizer.flash_attention_triton import FlashAttention

        return FlashAttention().run(q=q, k=k, v=v, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p))

    def _run_mps(self, *, q: Tensor, k: Tensor, v: Tensor, causal: bool, scale: float, dropout_p: float) -> Tensor:
        self._require(bool(KERNELS.mps_available), msg="MPS attention training requires torch.backends.mps to be available.")
        from caramba.optimizer.metal.attention_training import MetalAttentionTraining

        return MetalAttentionTraining().run(
            q=q,
            k=k,
            v=v,
            causal=bool(causal),
            scale=float(scale),
            dropout_p=float(dropout_p),
        )

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
        self._validate_common(q=q, k=k, v=v, attn_mask=attn_mask, dropout_p=float(dropout_p))
        if q.device.type == "cuda":
            return self._run_cuda(q=q, k=k, v=v, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p))
        if q.device.type == "mps":
            return self._run_mps(q=q, k=k, v=v, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p))
        raise RuntimeError(f"AttentionTraining: unsupported device.type={q.device.type!r}.")

