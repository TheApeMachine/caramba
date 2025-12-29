"""Linear algebra helpers (named techniques) for high-performance init/training.

This module exists to keep heavyweight math (SVD approximations, etc.) out of
higher-level code like model loaders and trainers.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = [
    "randomized_svd",
]


def _stable_int_hash(s: str) -> int:
    # Stable 32-bit FNV-1a hash for deterministic sampling across runs.
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def randomized_svd(
    A: Tensor,
    *,
    rank: int,
    n_iter: int = 2,
    oversample: int = 8,
    seed: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute an approximate truncated SVD using randomized range finding.

    Returns (U, S, Vh) with shapes:
      U:  (m, r)
      S:  (r,)
      Vh: (r, n)

    Notes:
    - Intended for *initialization* tasks (e.g. attention surgery) where a
      fast approximation is preferred over exact decomposition.
    - Works best when A has a decaying spectrum and r << min(m, n).
    """
    if A.dim() != 2:
        raise ValueError("randomized_svd expects a 2D matrix")

    m, n = int(A.shape[0]), int(A.shape[1])
    r = int(rank)
    if r <= 0 or m <= 0 or n <= 0:
        raise ValueError("Invalid shapes/rank for randomized_svd")

    r = min(r, min(m, n))
    k = min(min(m, n), r + max(0, int(oversample)))

    # Random probe (deterministic if seed provided).
    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=A.device)
        gen.manual_seed(_stable_int_hash(str(seed)))

    Omega = torch.randn((n, k), device=A.device, dtype=A.dtype, generator=gen)

    # Range finding with optional power iterations.
    Y = A @ Omega  # (m, k)
    iters = max(0, int(n_iter))
    for _ in range(iters):
        Y = A @ (A.transpose(0, 1) @ Y)

    # Orthonormal basis for the range of A.
    #
    # NOTE: `torch.linalg.qr` and `torch.linalg.svd` currently fall back to CPU on MPS.
    # We explicitly move the calculation to CPU to silence the runtime warning and
    # make the host-device sync explicit.
    if Y.device.type == "mps":
        # Optimization: Transfer fp16 to CPU first, then cast to float32.
        # This reduces PCIe/bus bandwidth by 2x compared to casting on GPU then transferring.
        Uy, _Sy, _Vhy = torch.linalg.svd(Y.cpu().float(), full_matrices=False)
        Uy = Uy.to(device=Y.device)
    else:
        Uy, _Sy, _Vhy = torch.linalg.svd(Y.float(), full_matrices=False)

    Q = Uy.to(dtype=Y.dtype)  # (m, k)

    # Small SVD on projected matrix.
    B = Q.transpose(0, 1) @ A  # (k, n)

    if B.device.type == "mps":
        U_hat, S, Vh = torch.linalg.svd(B.cpu().float(), full_matrices=False)
        U_hat = U_hat.to(device=B.device)
        S = S.to(device=B.device)
        Vh = Vh.to(device=B.device)
    else:
        U_hat, S, Vh = torch.linalg.svd(B.float(), full_matrices=False)

    U = Q @ U_hat.to(dtype=Q.dtype)  # (m, k)
    Vh = Vh.to(dtype=Q.dtype)
    S = S.to(dtype=torch.float32)

    return U[:, :r], S[:r], Vh[:r, :]

