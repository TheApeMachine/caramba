#!/usr/bin/env python3
"""
CCL-MNIST: Constructive Compression Learning on MNIST (no gradient descent, no pretrained models).

Pipeline:
  1) Learn a patch "codec" with k-means (vector quantization).
  2) Tokenize images into a 2D grid of discrete patch IDs.
  3) Train class-conditional context models by counting:
       p(token | left, up, upleft) with Dirichlet smoothing and interpolation backoff.
  4) Classify by minimum codelength (maximum log-likelihood).
  5) Generate images by sampling tokens autoregressively and decoding patch IDs back to pixels.

This is intentionally simple and hackable. The main performance knobs are:
  - K (codebook size)
  - patch size / stride (token grid resolution)
  - context templates / interpolation weights
  - smoothing alpha

Expected outcome: you should get a working classifier + class-conditional sampler that produces
MNIST-like digits, entirely without gradient descent.

Example:
  python ccl_mnist.py --K 256 --patch 4 --stride 2 --alpha 0.5 --out runs/ccl1

Dependencies:
  pip install numpy matplotlib tqdm torch torchvision scikit-learn
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_mnist(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_train: (60000, 28, 28) float32 in [0,1]
      y_train: (60000,) int64
      x_test:  (10000, 28, 28) float32 in [0,1]
      y_test:  (10000,) int64
    """
    try:
        import torch
        from torchvision import datasets
    except Exception as e:
        raise RuntimeError("Need torch + torchvision. Install: pip install torch torchvision") from e

    root = Path(root)
    ensure_dir(root)

    ds_train = datasets.MNIST(root=str(root), train=True, download=True)
    ds_test = datasets.MNIST(root=str(root), train=False, download=True)

    x_train = ds_train.data.numpy().astype(np.float32) / 255.0
    y_train = ds_train.targets.numpy().astype(np.int64)
    x_test = ds_test.data.numpy().astype(np.float32) / 255.0
    y_test = ds_test.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def sample_random_patches(
    images: np.ndarray,
    n_patches: int,
    patch: int,
    stride: int,
    seed: int,
) -> np.ndarray:
    """
    Randomly samples patches from images.
    Uses positions aligned to the token grid (stride steps) to match later tokenization.
    """
    rng = np.random.default_rng(seed)
    H, W = images.shape[1], images.shape[2]
    out_h = (H - patch) // stride + 1
    out_w = (W - patch) // stride + 1

    patches = np.empty((n_patches, patch * patch), dtype=np.float32)
    for i in tqdm(range(n_patches), desc="Sampling patches"):
        idx = int(rng.integers(0, images.shape[0]))
        pi = int(rng.integers(0, out_h))
        pj = int(rng.integers(0, out_w))
        x = pi * stride
        y = pj * stride
        p = images[idx, x:x + patch, y:y + patch]
        patches[i] = p.reshape(-1)
    return patches


def fit_kmeans_codebook(
    patches: np.ndarray,
    K: int,
    seed: int,
    batch_size: int = 4096,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Learns K patch prototypes (centroids). Returns centroids (K, D) float32.
    Uses MiniBatchKMeans for speed.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[reportMissingImports]
    except Exception as e:
        raise RuntimeError("Need scikit-learn. Install: pip install scikit-learn") from e

    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        batch_size=batch_size,
        n_init=1,  # type: ignore[reportArgumentType]  # sklearn accepts int, type stubs are incorrect
        max_iter=max_iter,
        verbose=0,
    )
    km.fit(patches)
    cluster_centers = getattr(km, "cluster_centers_", None)
    if cluster_centers is None:
        raise RuntimeError("KMeans clustering failed")
    centers = np.asarray(cluster_centers, dtype=np.float32)
    return centers


def batch_tokenize_images(
    images: np.ndarray,
    centroids: np.ndarray,
    patch: int,
    stride: int,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Vector-quantizes images into token grids.

    images: (N, 28, 28) float32
    centroids: (K, patch*patch) float32
    returns tokens: (N, out_h, out_w) uint16
    """
    N, H, W = images.shape
    K, D = centroids.shape
    assert D == patch * patch

    out_h = (H - patch) // stride + 1
    out_w = (W - patch) // stride + 1

    tokens = np.empty((N, out_h, out_w), dtype=np.uint16)

    # Precompute centroid norms for fast squared distance.
    C = centroids.astype(np.float32)
    C_T = C.T  # (D, K)
    c_norm = np.sum(C * C, axis=1, dtype=np.float32).reshape(1, K)  # (1, K)

    for start in tqdm(range(0, N, batch_size), desc="Tokenizing"):
        end = min(N, start + batch_size)
        B = end - start
        batch = images[start:end]  # (B, H, W)

        # Strided window view: (B, out_h, out_w, patch, patch)
        windows = as_strided(
            batch,
            shape=(B, out_h, out_w, patch, patch),
            strides=(
                batch.strides[0],
                batch.strides[1] * stride,
                batch.strides[2] * stride,
                batch.strides[1],
                batch.strides[2],
            ),
        )

        P = windows.reshape(B * out_h * out_w, D).astype(np.float32, copy=True)

        # Squared distances to centroids: ||x||^2 + ||c||^2 - 2 x·c
        x_norm = np.sum(P * P, axis=1, dtype=np.float32).reshape(-1, 1)  # (Npatch, 1)
        dist = x_norm + c_norm - 2.0 * (P @ C_T)  # (Npatch, K)

        tok = np.argmin(dist, axis=1).astype(np.uint16)
        tokens[start:end] = tok.reshape(B, out_h, out_w)

    return tokens


def decode_tokens_to_image(
    token_grid: np.ndarray,
    centroids: np.ndarray,
    patch: int,
    stride: int,
) -> np.ndarray:
    """
    Decodes a token grid back to a 28x28-ish image by overlap-averaging patch prototypes.
    """
    Ht, Wt = token_grid.shape
    H = (Ht - 1) * stride + patch
    W = (Wt - 1) * stride + patch

    img = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)

    for i in range(Ht):
        x = i * stride
        for j in range(Wt):
            y = j * stride
            t = int(token_grid[i, j])
            p = centroids[t].reshape(patch, patch)
            img[x:x + patch, y:y + patch] += p
            wgt[x:x + patch, y:y + patch] += 1.0

    img = img / np.maximum(wgt, 1e-6)
    img = np.clip(img, 0.0, 1.0)
    return img


# -----------------------------
# Context models (counting)
# -----------------------------

@dataclass
class SparseCounts:
    total: int
    counts: dict[int, int]  # token -> count


@dataclass
class ClassModel:
    K: int
    base: int
    alpha: float
    use_full: bool

    # Unigram counts (dense)
    uni: np.ndarray  # (K,) int64
    uni_total: int

    # Context tables (sparse)
    # mid context: (left1, up1)
    mid: dict[int, SparseCounts]
    # full context: (left1, up1, upleft)
    full: dict[int, SparseCounts] | None


def _update_sparse(table: dict[int, SparseCounts], key: int, token: int) -> None:
    ent = table.get(key)
    if ent is None:
        table[key] = SparseCounts(total=1, counts={token: 1})
        return
    ent.total += 1
    ent.counts[token] = ent.counts.get(token, 0) + 1


def train_class_models(
    tokens: np.ndarray,
    labels: np.ndarray,
    K: int,
    alpha: float,
    use_full: bool = True,
) -> list[ClassModel]:
    """
    Trains 10 class-conditional models by counting token occurrences in contexts.
    tokens: (N, Ht, Wt) uint16
    labels: (N,) int64
    """
    num_classes = int(labels.max()) + 1
    assert num_classes == 10, "This script assumes MNIST labels 0..9"

    base = K + 1
    BOS = K  # boundary token

    models: list[ClassModel] = []
    for _ in range(num_classes):
        models.append(
            ClassModel(
                K=K,
                base=base,
                alpha=alpha,
                use_full=use_full,
                uni=np.zeros(K, dtype=np.int64),
                uni_total=0,
                mid={},
                full={} if use_full else None,
            )
        )

    Ht, Wt = tokens.shape[1], tokens.shape[2]

    for grid, y in tqdm(zip(tokens, labels, strict=True), total=tokens.shape[0], desc="Training context models"):
        m = models[int(y)]

        # Unigram update using bincount (fast)
        flat = grid.reshape(-1).astype(np.int64, copy=False)
        m.uni += np.bincount(flat, minlength=K).astype(np.int64)

        mid = m.mid
        full = m.full

        # Update contexts
        for i in range(Ht):
            for j in range(Wt):
                t = int(grid[i, j])
                l1 = int(grid[i, j - 1]) if j > 0 else BOS
                u1 = int(grid[i - 1, j]) if i > 0 else BOS
                ul = int(grid[i - 1, j - 1]) if (i > 0 and j > 0) else BOS

                key_mid = l1 * base + u1
                _update_sparse(mid, key_mid, t)

                if use_full and full is not None:
                    key_full = key_mid * base + ul
                    _update_sparse(full, key_full, t)

    # finalize unigram totals
    for m in models:
        m.uni_total = int(m.uni.sum())

    return models


def _prob_sparse(ent: SparseCounts | None, token: int, alpha: float, alphaK: float, K: int) -> float:
    if ent is None:
        # alpha / (0 + alphaK) = 1/K
        return 1.0 / float(K)
    c = ent.counts.get(token, 0)
    return (c + alpha) / (ent.total + alphaK)


def loglik_image_under_model(
    model: ClassModel,
    grid: np.ndarray,
    lam_full: float,
    lam_mid: float,
    lam_uni: float,
) -> float:
    """
    Computes log p(grid) under a class model using interpolated contexts:
      p = lam_full p_full + lam_mid p_mid + lam_uni p_uni
    """
    K = model.K
    base = model.base
    alpha = model.alpha
    alphaK = alpha * K
    BOS = K

    Ht, Wt = grid.shape
    ll = 0.0

    mid = model.mid
    full = model.full

    uni = model.uni
    uni_total = model.uni_total
    denom_uni = uni_total + alphaK

    for i in range(Ht):
        for j in range(Wt):
            t = int(grid[i, j])
            l1 = int(grid[i, j - 1]) if j > 0 else BOS
            u1 = int(grid[i - 1, j]) if i > 0 else BOS
            ul = int(grid[i - 1, j - 1]) if (i > 0 and j > 0) else BOS

            key_mid = l1 * base + u1
            p_mid = _prob_sparse(mid.get(key_mid), t, alpha, alphaK, K)

            p_uni = (int(uni[t]) + alpha) / denom_uni

            if model.use_full and full is not None:
                key_full = key_mid * base + ul
                p_full = _prob_sparse(full.get(key_full), t, alpha, alphaK, K)
                p = lam_full * p_full + lam_mid * p_mid + lam_uni * p_uni
            else:
                p = lam_mid * p_mid + lam_uni * p_uni

            # Numeric guard
            if p <= 0.0:
                p = 1e-12
            ll += float(np.log(p))

    return ll


def predict(
    models: list[ClassModel],
    grid: np.ndarray,
    lam_full: float,
    lam_mid: float,
    lam_uni: float,
) -> int:
    scores = [
        loglik_image_under_model(m, grid, lam_full=lam_full, lam_mid=lam_mid, lam_uni=lam_uni)
        for m in models
    ]
    return int(np.argmax(scores))


# -----------------------------
# Sampling
# -----------------------------

def _sample_from_sparse_dirichlet_smoothed(
    ent: SparseCounts | None,
    rng: np.random.Generator,
    alphaK: float,
    K: int,
) -> int:
    """
    Sample token from Dirichlet-smoothed categorical:
      p(t) = (count(t) + alpha) / (total + alphaK)
    Efficient sampling without enumerating all K:
      With probability total/(total+alphaK): sample proportional to counts
      Otherwise: sample uniform over K (represents pseudo-count mass)
    """
    if ent is None or ent.total <= 0:
        return int(rng.integers(0, K))

    total = ent.total
    if rng.random() < (total / (total + alphaK)):
        # sample from observed counts only
        r = int(rng.integers(0, total))
        s = 0
        for tok, cnt in ent.counts.items():
            s += cnt
            if r < s:
                return int(tok)
        # fallback (shouldn't happen)
        return int(next(iter(ent.counts.keys())))
    else:
        return int(rng.integers(0, K))


def _sample_from_unigram(
    uni: np.ndarray,
    uni_total: int,
    rng: np.random.Generator,
    alphaK: float,
    K: int,
) -> int:
    if uni_total <= 0:
        return int(rng.integers(0, K))
    if rng.random() < (uni_total / (uni_total + alphaK)):
        # sample proportional to counts
        probs = (uni / uni_total).astype(np.float64, copy=False)
        return int(rng.choice(K, p=probs))
    else:
        return int(rng.integers(0, K))


def sample_token_grid(
    model: ClassModel,
    Ht: int,
    Wt: int,
    lam_full: float,
    lam_mid: float,
    lam_uni: float,
    seed: int,
) -> np.ndarray:
    """
    Autoregressively samples a token grid using mixture-of-contexts sampling.
    """
    rng = np.random.default_rng(seed)

    K = model.K
    base = model.base
    alpha = model.alpha
    alphaK = alpha * K
    BOS = K

    grid = np.empty((Ht, Wt), dtype=np.uint16)

    mid = model.mid
    full = model.full
    uni = model.uni
    uni_total = model.uni_total

    if not model.use_full:
        # renormalize weights for mid+uni
        s = lam_mid + lam_uni
        lam_mid2 = lam_mid / s
        lam_full2 = 0.0
    else:
        lam_full2, lam_mid2 = lam_full, lam_mid

    for i in range(Ht):
        for j in range(Wt):
            l1 = int(grid[i, j - 1]) if j > 0 else BOS
            u1 = int(grid[i - 1, j]) if i > 0 else BOS
            ul = int(grid[i - 1, j - 1]) if (i > 0 and j > 0) else BOS

            key_mid = l1 * base + u1
            ent_mid = mid.get(key_mid)

            r = rng.random()
            if model.use_full and full is not None and r < lam_full2:
                key_full = key_mid * base + ul
                ent_full = full.get(key_full)
                t = _sample_from_sparse_dirichlet_smoothed(ent_full, rng, alphaK, K)
            elif r < (lam_full2 + lam_mid2):
                t = _sample_from_sparse_dirichlet_smoothed(ent_mid, rng, alphaK, K)
            else:
                t = _sample_from_unigram(uni, uni_total, rng, alphaK, K)

            grid[i, j] = np.uint16(t)

    return grid


# -----------------------------
# Main script
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/ccl_mnist")
    ap.add_argument("--data", type=str, default="data/mnist")

    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--use_full", action="store_true", help="Use (left,up,upleft) full context in addition to (left,up).")

    ap.add_argument("--lam_full", type=float, default=0.55)
    ap.add_argument("--lam_mid", type=float, default=0.35)
    ap.add_argument("--lam_uni", type=float, default=0.10)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--sample_patches", type=int, default=200_000)
    ap.add_argument("--kmeans_max_iter", type=int, default=200)
    ap.add_argument("--tok_batch", type=int, default=256)

    ap.add_argument("--max_train", type=int, default=60000, help="Use fewer training images for faster iteration.")
    ap.add_argument("--max_test", type=int, default=10000)

    ap.add_argument("--n_gen_per_class", type=int, default=12, help="How many samples to generate per class.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Save config
    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")

    x_train, y_train, x_test, y_test = load_mnist(Path(args.data))

    if args.max_train < len(x_train):
        x_train = x_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_test < len(x_test):
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    # 1) Codec learning (k-means)
    patches = sample_random_patches(
        images=x_train,
        n_patches=int(args.sample_patches),
        patch=int(args.patch),
        stride=int(args.stride),
        seed=int(args.seed),
    )
    centroids = fit_kmeans_codebook(
        patches=patches,
        K=int(args.K),
        seed=int(args.seed),
        max_iter=int(args.kmeans_max_iter),
    )
    np.save(out_dir / "centroids.npy", centroids)

    # 2) Tokenize train/test
    train_tokens = batch_tokenize_images(
        images=x_train,
        centroids=centroids,
        patch=int(args.patch),
        stride=int(args.stride),
        batch_size=int(args.tok_batch),
    )
    test_tokens = batch_tokenize_images(
        images=x_test,
        centroids=centroids,
        patch=int(args.patch),
        stride=int(args.stride),
        batch_size=int(args.tok_batch),
    )
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "test_tokens.npy", test_tokens)

    # 3) Train context models
    models = train_class_models(
        tokens=train_tokens,
        labels=y_train,
        K=int(args.K),
        alpha=float(args.alpha),
        use_full=bool(args.use_full),
    )
    with open(out_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 4) Evaluate
    lam_full = float(args.lam_full)
    lam_mid = float(args.lam_mid)
    lam_uni = float(args.lam_uni)

    correct = 0
    for grid, y in tqdm(zip(test_tokens, y_test, strict=True), total=test_tokens.shape[0], desc="Evaluating"):
        yhat = predict(models, grid, lam_full=lam_full, lam_mid=lam_mid, lam_uni=lam_uni)
        correct += int(yhat == int(y))
    acc = correct / float(test_tokens.shape[0])

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"test_accuracy={acc:.6f}\n")
        f.write(f"test_correct={correct}\n")
        f.write(f"test_total={int(test_tokens.shape[0])}\n")

    print(f"Test accuracy: {acc:.4f}")

    # 5) Generate samples and save a grid image
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Need matplotlib. Install: pip install matplotlib") from e

    # token grid dimensions
    Ht, Wt = train_tokens.shape[1], train_tokens.shape[2]
    n_per = int(args.n_gen_per_class)

    # Make a big mosaic: 10 rows (classes) x n_per columns
    fig_w = max(8, int(n_per * 1.2))
    fig_h = 10
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    plot_idx = 1
    for c in range(10):
        m = models[c]
        for k in range(n_per):
            grid = sample_token_grid(
                m,
                Ht=Ht,
                Wt=Wt,
                lam_full=lam_full,
                lam_mid=lam_mid,
                lam_uni=lam_uni,
                seed=int(args.seed + 1000 + c * 100 + k),
            )
            img = decode_tokens_to_image(grid, centroids, patch=int(args.patch), stride=int(args.stride))

            ax = fig.add_subplot(10, n_per, plot_idx)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(str(c), rotation=0, labelpad=10)
            plot_idx += 1

    plt.tight_layout(pad=0.2)
    fig_path = out_dir / "generated_grid.png"
    plt.savefig(fig_path)
    print(f"Saved samples to: {fig_path}")

    # Also save per-class sample grids as .npy if you want to inspect token grids later
    print("Done.")


if __name__ == "__main__":
    main()
