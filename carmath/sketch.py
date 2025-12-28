"""Sketching utilities for cheap per-tensor statistics.

This module exists to *name* and centralize common sketch-based math so higher-level
systems (like nowcasting) don't devolve into scattered index arithmetic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from optimizer.triton_runtime import TRITON_AVAILABLE

# Optional Triton bindings (populated only when available)
triton = None  # type: ignore[assignment]
tl = None  # type: ignore[assignment]
_sketch_dot5_indexed_kernel = None  # type: ignore[assignment]
_sketch_dot5_contiguous_kernel = None  # type: ignore[assignment]

__all__ = [
    "stable_int_hash",
    "stride_sketch_indices",
    "sketch_dot5",
]


def stable_int_hash(s: str) -> int:
    """Stable 32-bit FNV-1a hash for deterministic sampling across runs."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def stride_sketch_indices(
    numel: int,
    k: int,
    *,
    seed: str,
    device: torch.device,
    hashed_start: bool = True,
) -> Tensor:
    """Deterministic strided subsample indices into a flattened tensor.

    Uses an O(k) pattern, avoiding randperm(numel) which is O(numel).
    """
    n = int(numel)
    kk = int(k)
    if n <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if kk <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if kk >= n:
        return torch.arange(n, device=device, dtype=torch.long)

    stride = max(1, n // kk)
    start = (stable_int_hash(seed) % stride) if hashed_start else 0
    return (start + torch.arange(kk, device=device, dtype=torch.long) * stride) % n


def _sketch_dot5_torch(
    w: Tensor,
    w_prev: Tensor,
    u: Tensor,
    g: Tensor | None,
    idx: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Torch fallback for 5 sketch dot-products (float32 scalars on device)."""
    if idx is None:
        wv = w.reshape(-1)
        wpv = w_prev.reshape(-1)
        uv = u.reshape(-1)
        gv = g.reshape(-1) if g is not None else None
    else:
        wv = w.reshape(-1).index_select(0, idx)
        wpv = w_prev.reshape(-1).index_select(0, idx)
        uv = u.reshape(-1).index_select(0, idx)
        gv = g.reshape(-1).index_select(0, idx) if g is not None else None

    uv32 = uv.float()
    t32 = (wv.float() - wpv.float())

    uu = torch.dot(uv32, uv32)
    tt = torch.dot(t32, t32)
    ut = torch.dot(uv32, t32)

    if gv is None:
        vv = uu.new_zeros(())
        uv_dot = uu.new_zeros(())
    else:
        gv32 = gv.float()
        vv = torch.dot(gv32, gv32)
        uv_dot = torch.dot(uv32, gv32)

    return uu, tt, ut, vv, uv_dot


if not TYPE_CHECKING and TRITON_AVAILABLE:
    try:
        import triton as _triton  # type: ignore
        import triton.language as _tl  # type: ignore
    except (ImportError, ModuleNotFoundError):
        triton = None  # type: ignore[assignment]
        tl = None  # type: ignore[assignment]
    else:
        triton = _triton  # type: ignore[assignment]
        tl = _tl  # type: ignore[assignment]

        @triton.jit  # type: ignore[union-attr]
        def _sketch_dot5_indexed_kernel(
            W_ptr,
            WP_ptr,
            U_ptr,
            G_ptr,  # may be null (handled via has_grad)
            IDX_ptr,
            OUT_ptr,  # (5,) float32 scalars
            K: tl.constexpr,
            has_grad: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < K

            idx = tl.load(IDX_ptr + offs, mask=mask, other=0).to(tl.int64)
            w = tl.load(W_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            wp = tl.load(WP_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            t = w - wp

            uu = tl.sum(u * u, axis=0)
            tt = tl.sum(t * t, axis=0)
            ut = tl.sum(u * t, axis=0)

            if has_grad:
                g = tl.load(G_ptr + idx, mask=mask, other=0.0).to(tl.float32)
                vv = tl.sum(g * g, axis=0)
                uv = tl.sum(u * g, axis=0)
            else:
                vv = 0.0
                uv = 0.0

            tl.atomic_add(OUT_ptr + 0, uu)
            tl.atomic_add(OUT_ptr + 1, tt)
            tl.atomic_add(OUT_ptr + 2, ut)
            tl.atomic_add(OUT_ptr + 3, vv)
            tl.atomic_add(OUT_ptr + 4, uv)

        @triton.jit  # type: ignore[union-attr]
        def _sketch_dot5_contiguous_kernel(
            W_ptr,
            WP_ptr,
            U_ptr,
            G_ptr,  # may be null (handled via has_grad)
            OUT_ptr,  # (5,)
            N: tl.constexpr,
            has_grad: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N

            w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            wp = tl.load(WP_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            t = w - wp

            uu = tl.sum(u * u, axis=0)
            tt = tl.sum(t * t, axis=0)
            ut = tl.sum(u * t, axis=0)

            if has_grad:
                g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                vv = tl.sum(g * g, axis=0)
                uv = tl.sum(u * g, axis=0)
            else:
                vv = 0.0
                uv = 0.0

            tl.atomic_add(OUT_ptr + 0, uu)
            tl.atomic_add(OUT_ptr + 1, tt)
            tl.atomic_add(OUT_ptr + 2, ut)
            tl.atomic_add(OUT_ptr + 3, vv)
            tl.atomic_add(OUT_ptr + 4, uv)


def sketch_dot5(
    w: Tensor,
    w_prev: Tensor,
    u: Tensor,
    g: Tensor | None,
    idx: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute 5 dot-products used by sketch-based relative loss.

    Returns float32 scalars (0-d tensors) on the same device:
      uu = ||u||^2
      tt = ||t||^2, where t = w - w_prev
      ut = u·t
      vv = ||g||^2 (0 if g is None)
      uv = u·g      (0 if g is None)
    """
    # Fast path: Triton on CUDA.
    if (
        TRITON_AVAILABLE
        and triton is not None
        and _sketch_dot5_contiguous_kernel is not None
        and _sketch_dot5_indexed_kernel is not None
        and w.is_cuda
        and w_prev.is_cuda
        and u.is_cuda
        and (g is None or g.is_cuda)
    ):
        # Triton kernels assume contiguous 1D pointers.
        # If we can't get a view without copying, fall back.
        try:
            wv = w.reshape(-1)
            wpv = w_prev.reshape(-1)
            uv = u.reshape(-1)
            gv = g.reshape(-1) if g is not None else None
        except Exception:
            return _sketch_dot5_torch(w, w_prev, u, g, idx)

        if not (wv.is_contiguous() and wpv.is_contiguous() and uv.is_contiguous()):
            return _sketch_dot5_torch(w, w_prev, u, g, idx)
        if gv is not None and not gv.is_contiguous():
            return _sketch_dot5_torch(w, w_prev, u, g, idx)

        out = torch.zeros((5,), device=wv.device, dtype=torch.float32)
        has_grad = gv is not None

        # Choose a reasonable block size.
        BLOCK = 1024

        if idx is None:
            n = int(wv.numel())
            grid = (triton.cdiv(n, BLOCK),)  # type: ignore[union-attr]
            _sketch_dot5_contiguous_kernel[grid](  # type: ignore[index]
                wv,
                wpv,
                uv,
                gv if gv is not None else wv,  # dummy ptr when has_grad=False
                out,
                n,
                has_grad=has_grad,
                BLOCK=BLOCK,
                num_warps=4,
            )
        else:
            k = int(idx.numel())
            if k <= 0:
                z = out[0]
                return z, z, z, z, z
            if not idx.is_cuda:
                return _sketch_dot5_torch(w, w_prev, u, g, idx)
            if not idx.is_contiguous():
                idx = idx.contiguous()
            grid = (triton.cdiv(k, BLOCK),)  # type: ignore[union-attr]
            _sketch_dot5_indexed_kernel[grid](  # type: ignore[index]
                wv,
                wpv,
                uv,
                gv if gv is not None else wv,
                idx,
                out,
                K=k,
                has_grad=has_grad,
                BLOCK=BLOCK,
                num_warps=4,
            )

        return out[0], out[1], out[2], out[3], out[4]

    return _sketch_dot5_torch(w, w_prev, u, g, idx)

