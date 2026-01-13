"""Attention training kernels and dispatch.

Caramba's attention policy:
- CUDA training uses custom Triton FlashAttention (forward+backward).
- MPS training uses custom Metal fused attention (forward+backward).

This module provides a single composable object for attention execution so layers
don't need to embed backend-specific logic.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._dynamo import is_compiling as _dynamo_is_compiling

from caramba.kernel.kernel_registry import KERNELS
from caramba.console import logger
from caramba.kernel.triton.flash_attention_triton import FlashAttention
from caramba.kernel.metal.attention_training import MetalAttentionTraining


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

    # Cache of the fastest CUDA backend per (dtype, T, D, causal).
    #
    # Why:
    # - PyTorch SDPA on CUDA can dispatch to FlashAttention2 / efficient kernels that may
    #   outperform our Triton implementation for some shapes/devices.
    # - We benchmark once per shape and then deterministically reuse the faster backend.
    _cuda_backend_cache: dict[tuple[torch.dtype, int, int, bool], str] = {}

    def _bench_cuda_fwd_bwd_ms(
        self,
        *,
        fn_name: str,
        fn: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> float:
        """Benchmark a CUDA attention backend (forward+backward) in milliseconds."""
        if q.device.type != "cuda":
            raise RuntimeError("CUDA benchmark requires q on CUDA")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise RuntimeError("CUDA benchmark requires contiguous q/k/v")

        # Reuse the same leaf tensors across iterations; each iteration builds a fresh graph.
        q1 = q.detach().clone().requires_grad_(True)
        k1 = k.detach().clone().requires_grad_(True)
        v1 = v.detach().clone().requires_grad_(True)

        def step() -> None:
            q1.grad = None
            k1.grad = None
            v1.grad = None
            out = fn(q1, k1, v1)
            # Use a cheap scalar loss.
            loss = out.float().mean()
            loss.backward()

        # Warmup.
        for _ in range(2):
            step()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 5
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end)) / float(iters)
        logger.info(f"AttentionTraining CUDA bench {fn_name}: {ms:.3f} ms (fwd+bwd)")
        return ms

    def _select_cuda_backend(self, *, q: Tensor, k: Tensor, v: Tensor, causal: bool, scale: float, dropout_p: float) -> str:
        # Only benchmark the deterministic, dropout-free case.
        if float(dropout_p) != 0.0:
            return "triton"

        # torch.compile/inductor can crash when we run profiler/event timing + backward
        # during the compile trace. In compile mode, just use PyTorch SDPA (which will
        # dispatch to FlashAttention2 on A100) and defer any benchmarking.
        if bool(_dynamo_is_compiling()):
            return "sdpa"

        # Do not run backward-based benchmarking under `torch.no_grad()`.
        # This path is exercised during reproducibility IO-shape export (no_grad),
        # where we only need a forward pass. Benchmarking is deferred until the
        # first grad-enabled training step.
        if not torch.is_grad_enabled():
            return "triton"

        T = int(q.shape[2])
        D = int(q.shape[3])
        key = (q.dtype, T, D, bool(causal))
        cached = self._cuda_backend_cache.get(key, None)
        if isinstance(cached, str):
            return cached

        def triton_fn(qi: Tensor, ki: Tensor, vi: Tensor) -> Tensor:
            return FlashAttention().run(q=qi, k=ki, v=vi, causal=bool(causal), scale=float(scale), dropout_p=0.0)

        def sdpa_fn(qi: Tensor, ki: Tensor, vi: Tensor) -> Tensor:
            return F.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=bool(causal),
                scale=float(scale),
            )

        ms_triton = self._bench_cuda_fwd_bwd_ms(fn_name="triton_flash", fn=triton_fn, q=q, k=k, v=v)
        ms_sdpa = self._bench_cuda_fwd_bwd_ms(fn_name="pytorch_sdpa", fn=sdpa_fn, q=q, k=k, v=v)
        backend = "sdpa" if ms_sdpa < ms_triton else "triton"
        self._cuda_backend_cache[key] = backend
        logger.info(
            f"AttentionTraining selected CUDA backend={backend} (sdpa_ms={ms_sdpa:.3f}, triton_ms={ms_triton:.3f}) "
            f"for dtype={q.dtype} T={T} D={D} causal={bool(causal)}"
        )
        return backend

    def _run_cuda(self, *, q: Tensor, k: Tensor, v: Tensor, causal: bool, scale: float, dropout_p: float) -> Tensor:
        self._require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="CUDA attention training requires Triton kernels validated at startup.",
        )

        backend = self._select_cuda_backend(
            q=q,
            k=k,
            v=v,
            causal=bool(causal),
            scale=float(scale),
            dropout_p=float(dropout_p),
        )

        if backend == "sdpa":
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=float(dropout_p),
                is_causal=bool(causal),
                scale=float(scale),
            )

        return FlashAttention().run(q=q, k=k, v=v, causal=bool(causal), scale=float(scale), dropout_p=float(dropout_p))

    def _run_mps(self, *, q: Tensor, k: Tensor, v: Tensor, causal: bool, scale: float, dropout_p: float) -> Tensor:
        self._require(
            cond=bool(KERNELS.mps_available),
            msg="MPS attention training requires torch.backends.mps to be available."
        )

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

