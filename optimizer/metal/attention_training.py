"""Metal fused attention training (forward + backward).

This module provides a custom full-sequence attention kernel for Apple MPS:
- Forward computes output and per-query log-sum-exp (LSE)
- Backward recomputes softmax probabilities from (Q,K,LSE) and accumulates dQ/dK/dV

This replaces PyTorch SDPA for MPS training under caramba's kernel policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from caramba.optimizer.runtime import METAL_SUPPORTED

from .attention_jit import load_caramba_metal_attention_ops


def metal_attention_training_available() -> bool:
    """Whether the runtime is capable of using the Metal attention training path."""
    return bool(METAL_SUPPORTED and torch.backends.mps.is_available())


@dataclass(frozen=True, slots=True)
class _AttnMeta:
    causal: bool
    scale: float
    dropout_p: float
    seed: int


class _MetalAttnTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        scale: float,
        dropout_p: float,
        seed: int,
    ) -> Tensor:
        if q.device.type != "mps":
            raise RuntimeError("MetalAttentionTraining requires device.type == 'mps'")
        if q.dtype != torch.float16:
            raise RuntimeError(f"MetalAttentionTraining requires fp16 inputs on MPS (got dtype={q.dtype})")

        q2 = q.contiguous()
        k2 = k.contiguous()
        v2 = v.contiguous()

        if q2.shape != k2.shape or q2.shape != v2.shape:
            raise RuntimeError("MetalAttentionTraining requires q/k/v shapes to match (B,H,T,D)")
        if q2.ndim != 4:
            raise RuntimeError(f"MetalAttentionTraining expects q/k/v shape (B,H,T,D), got shape={tuple(q2.shape)}")

        ops = load_caramba_metal_attention_ops(verbose=False)
        out, lse = ops.attn_train_fwd(q2, k2, v2, float(scale), bool(causal), float(dropout_p), int(seed))

        meta = _AttnMeta(causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p), seed=int(seed))
        ctx.meta = meta  # type: ignore[attr-defined]
        ctx.save_for_backward(q2, k2, v2, out, lse)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_out: Tensor,
    ) -> tuple[Tensor | None, ...]:
        q, k, v, out, lse = ctx.saved_tensors
        meta: _AttnMeta = ctx.meta  # type: ignore[attr-defined]

        ops = load_caramba_metal_attention_ops(verbose=False)
        dq, dk, dv = ops.attn_train_bwd(
            q,
            k,
            v,
            out,
            lse,
            grad_out.contiguous(),
            float(meta.scale),
            bool(meta.causal),
            float(meta.dropout_p),
            int(meta.seed),
        )
        return (dq, dk, dv, None, None, None, None)


class MetalAttentionTraining:
    """Metal attention training kernel.

    Use `.run(...)` to execute a fused full-sequence attention on MPS.
    """

    def run(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        scale: float,
        dropout_p: float = 0.0,
        seed: int | None = None,
        verbose_build: bool = False,
    ) -> Tensor:
        """Run fused attention on MPS.

        Args:
            q/k/v: (B,H,T,D) fp16 on MPS, contiguous
            causal: whether to apply causal masking
            scale: scaling applied to QK^T
            dropout_p: dropout probability applied to attention weights
            seed: seed used to regenerate dropout mask in backward
            verbose_build: if True, verbose extension build output
        """
        if not metal_attention_training_available():
            raise RuntimeError("MetalAttentionTraining is unavailable on this runtime")
        if verbose_build:
            _ = load_caramba_metal_attention_ops(verbose=True)

        seed_i = int(torch.seed()) if seed is None else int(seed)
        y = _MetalAttnTrainFn.apply(q, k, v, bool(causal), float(scale), float(dropout_p), seed_i)
        if not isinstance(y, torch.Tensor):
            raise TypeError("MetalAttentionTraining returned a non-tensor output")
        return y

