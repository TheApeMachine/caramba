from __future__ import annotations

"""Patch-based vector quantization (VQ) codec for CCL.

This is intentionally lightweight and dependency-free (no scikit-learn).
It implements a simple minibatch k-means suitable for learning a patch codebook.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _ensure_nchw(x: np.ndarray) -> np.ndarray:
    """Coerce image batch to NCHW float32."""
    if x.ndim == 3:
        # NHW -> N1HW
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Expected images with shape (N,H,W) or (N,C,H,W), got {x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _random_patches_aligned(
    images_nchw: np.ndarray,
    *,
    n_patches: int,
    patch: int,
    stride: int,
    seed: int,
) -> np.ndarray:
    """Sample flattened patches aligned to the stride grid."""
    rng = np.random.default_rng(int(seed))
    x = _ensure_nchw(images_nchw)
    n, c, h, w = x.shape
    if h < patch or w < patch:
        raise ValueError(f"patch={patch} must fit within image HxW={h}x{w}")
    out_h = (h - patch) // stride + 1
    out_w = (w - patch) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid patch/stride for given image size")

    d = int(c * patch * patch)
    patches = np.empty((int(n_patches), d), dtype=np.float32)
    for i in range(int(n_patches)):
        idx = int(rng.integers(0, n))
        pi = int(rng.integers(0, out_h))
        pj = int(rng.integers(0, out_w))
        x0 = pi * stride
        y0 = pj * stride
        p = x[idx, :, x0 : x0 + patch, y0 : y0 + patch]
        patches[i] = p.reshape(-1).astype(np.float32, copy=False)
    return patches


def _minibatch_kmeans(
    x: np.ndarray,
    *,
    k: int,
    seed: int,
    batch_size: int = 4096,
    max_iter: int = 200,
    init: str = "sample",
) -> np.ndarray:
    """Simple minibatch k-means (returns centers with shape (k, d)).

    Notes:
    - This is not a perfect replica of scikit-learn's MiniBatchKMeans.
    - It is stable and deterministic for a fixed seed.
    """
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x as (N,D), got {x.shape}")
    n, d = x.shape
    if n <= 0:
        raise ValueError("x is empty")
    if k <= 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")

    if init == "sample":
        centers = x[rng.choice(n, size=int(k), replace=False)].copy()
    else:
        raise ValueError(f"Unknown init={init!r}")

    # Online update with per-center counts.
    counts = np.ones((int(k),), dtype=np.float32)
    centers = centers.astype(np.float32, copy=False)

    c_norm = np.sum(centers * centers, axis=1).astype(np.float32, copy=False)  # (k,)
    for _ in range(int(max_iter)):
        b = int(min(int(batch_size), n))
        batch = x[rng.integers(0, n, size=b)]
        # dist^2 = ||x||^2 + ||c||^2 - 2x·c
        x_norm = np.sum(batch * batch, axis=1).astype(np.float32, copy=False)[:, None]  # (b,1)
        dist = x_norm + c_norm[None, :] - 2.0 * (batch @ centers.T)  # (b,k)
        assign = np.argmin(dist, axis=1).astype(np.int64)
        # Update centers (streaming mean).
        for j in range(b):
            ci = int(assign[j])
            counts[ci] += 1.0
            eta = 1.0 / counts[ci]
            centers[ci] = (1.0 - eta) * centers[ci] + eta * batch[j]
        c_norm = np.sum(centers * centers, axis=1).astype(np.float32, copy=False)
    return centers.astype(np.float32, copy=False)


def _tokenize_batch(
    images_nchw: np.ndarray,
    centers: np.ndarray,
    *,
    patch: int,
    stride: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Tokenize images into a grid of patch IDs (N,Ht,Wt) int32."""
    x = _ensure_nchw(images_nchw)
    n, c, h, w = x.shape
    k, d = centers.shape
    if int(d) != int(c * patch * patch):
        raise ValueError("Center dimensionality does not match patch shape")
    out_h = (h - patch) // stride + 1
    out_w = (w - patch) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid patch/stride for given image size")

    tokens = np.empty((n, out_h, out_w), dtype=np.int32)
    c_f = centers.astype(np.float32, copy=False)
    c_t = c_f.T  # (d,k)
    c_norm = np.sum(c_f * c_f, axis=1).astype(np.float32, copy=False)[None, :]  # (1,k)

    # Strided view for NCHW: (B, C, out_h, out_w, patch, patch)
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        b = end - start
        batch = x[start:end]  # (B,C,H,W)
        # Create a strided view without copying patches.
        # Layout: channels are contiguous within a patch.
        windows = np.lib.stride_tricks.as_strided(
            batch,
            shape=(b, c, out_h, out_w, patch, patch),
            strides=(
                batch.strides[0],
                batch.strides[1],
                batch.strides[2] * stride,
                batch.strides[3] * stride,
                batch.strides[2],
                batch.strides[3],
            ),
        )
        p = windows.transpose(0, 2, 3, 1, 4, 5).reshape(b * out_h * out_w, d).astype(
            np.float32, copy=False
        )
        x_norm = np.sum(p * p, axis=1).astype(np.float32, copy=False)[:, None]  # (npatch,1)
        dist = x_norm + c_norm - 2.0 * (p @ c_t)  # (npatch,k)
        tok = np.argmin(dist, axis=1).astype(np.int32)
        tokens[start:end] = tok.reshape(b, out_h, out_w)
    return tokens


def _decode_grid(
    grid: np.ndarray,
    centers: np.ndarray,
    *,
    out_h: int,
    out_w: int,
    channels: int,
    patch: int,
    stride: int,
) -> np.ndarray:
    """Decode token grid back to an image by overlap-averaging prototypes (CHW)."""
    ht, wt = grid.shape
    h = (ht - 1) * stride + patch
    w = (wt - 1) * stride + patch
    img = np.zeros((channels, h, w), dtype=np.float32)
    wgt = np.zeros((1, h, w), dtype=np.float32)
    d = int(channels * patch * patch)
    if centers.shape[1] != d:
        raise ValueError("Centers do not match channels/patch")
    for i in range(int(ht)):
        x0 = i * stride
        for j in range(int(wt)):
            y0 = j * stride
            t = int(grid[i, j])
            p = centers[t].reshape(channels, patch, patch)
            img[:, x0 : x0 + patch, y0 : y0 + patch] += p
            wgt[:, x0 : x0 + patch, y0 : y0 + patch] += 1.0
    img = img / np.maximum(wgt, 1e-6)
    return np.clip(img, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class PatchKMeansVQ:
    """Patch VQ codec using minibatch k-means."""

    k: int
    patch: int
    stride: int
    seed: int = 0
    sample_patches: int = 200_000
    kmeans_batch_size: int = 4096
    kmeans_max_iter: int = 200

    def fit(self, images: np.ndarray) -> np.ndarray:
        images = _ensure_nchw(images)
        patches = _random_patches_aligned(
            images,
            n_patches=int(self.sample_patches),
            patch=int(self.patch),
            stride=int(self.stride),
            seed=int(self.seed),
        )
        return _minibatch_kmeans(
            patches,
            k=int(self.k),
            seed=int(self.seed),
            batch_size=int(self.kmeans_batch_size),
            max_iter=int(self.kmeans_max_iter),
        )

    def tokenize(self, images: np.ndarray, *, centers: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return _tokenize_batch(
            images,
            np.asarray(centers, dtype=np.float32),
            patch=int(self.patch),
            stride=int(self.stride),
            batch_size=int(batch_size),
        )

    def decode(self, grid: np.ndarray, *, centers: np.ndarray, channels: int) -> np.ndarray:
        # Convenience wrapper; out_h/out_w are inferred from grid.
        return _decode_grid(
            np.asarray(grid),
            np.asarray(centers, dtype=np.float32),
            out_h=int(grid.shape[0]),
            out_w=int(grid.shape[1]),
            channels=int(channels),
            patch=int(self.patch),
            stride=int(self.stride),
        )

