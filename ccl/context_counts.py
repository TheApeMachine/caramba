from __future__ import annotations

"""Sparse context-count models for CCL.

This generalizes the MNIST script's (left,up[,upleft]) templates into arbitrary
2D offset templates. It remains simple: count tables + Dirichlet smoothing with
mixture interpolation (contexts + unigram).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Offset2D = Tuple[int, int]  # (di, dj)


@dataclass
class SparseCounts:
    total: int
    counts: Dict[int, int]  # token -> count


@dataclass(frozen=True, slots=True)
class ContextTemplate:
    """A context template is an ordered list of offsets and an interpolation weight."""

    name: str
    offsets: tuple[Offset2D, ...]
    weight: float

    def key_base(self, token: int, base: int, pos: int) -> int:
        # Not used; kept for future expansion.
        return int(token) * (base**pos)


@dataclass
class ClassCountsModel:
    k: int
    alpha: float
    templates: list[ContextTemplate]  # excluding unigram
    unigram_weight: float
    # boundary token (BOS) is k
    base: int  # (k + 1)

    uni: np.ndarray  # (k,) int64
    uni_total: int
    tables: list[dict[int, SparseCounts]]  # per-template: key -> sparse counts


def _update_sparse(table: dict[int, SparseCounts], key: int, tok: int) -> None:
    ent = table.get(key)
    if ent is None:
        table[key] = SparseCounts(total=1, counts={int(tok): 1})
        return
    ent.total += 1
    ent.counts[int(tok)] = int(ent.counts.get(int(tok), 0)) + 1


def _ctx_key(
    grid: np.ndarray,
    *,
    i: int,
    j: int,
    offsets: Sequence[Offset2D],
    bos: int,
    base: int,
) -> int:
    """Encode a context as a single integer key via mixed radix base=(k+1)."""
    key = 0
    mul = 1
    ht, wt = int(grid.shape[0]), int(grid.shape[1])
    for di, dj in offsets:
        ii = int(i + di)
        jj = int(j + dj)
        if (ii < 0) or (jj < 0) or (ii >= ht) or (jj >= wt):
            v = int(bos)
        else:
            v = int(grid[ii, jj])
        key += int(v) * int(mul)
        mul *= int(base)
    return int(key)


def _prob_sparse(ent: Optional[SparseCounts], tok: int, *, alpha: float, alpha_k: float, k: int) -> float:
    if ent is None:
        return 1.0 / float(k)
    c = int(ent.counts.get(int(tok), 0))
    return float((c + alpha) / (ent.total + alpha_k))


def train_class_counts_models(
    token_grids: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    alpha: float,
    templates: Sequence[ContextTemplate],
    unigram_weight: float,
    num_classes: int | None = None,
) -> tuple[list[ClassCountsModel], dict[int, int]]:
    """Train class-conditional context-count models.

    Returns:
      - models: list length C (C inferred unless num_classes provided)
      - label_to_class: mapping from original label value -> contiguous class index
    """
    grids = np.asarray(token_grids)
    y = np.asarray(labels).astype(np.int64, copy=False)
    if grids.ndim != 3:
        raise ValueError(f"Expected token grids as (N,Ht,Wt), got {grids.shape}")
    if y.ndim != 1 or int(y.shape[0]) != int(grids.shape[0]):
        raise ValueError("labels must be (N,) aligned to token_grids")
    if int(k) <= 1:
        raise ValueError("k must be > 1")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be > 0")

    uniq = sorted({int(v) for v in y.tolist()})
    if num_classes is None:
        num_classes = int(len(uniq))
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be > 0")

    # Map labels to contiguous [0..C-1] to keep models compact.
    label_to_class: dict[int, int] = {}
    for i, lab in enumerate(uniq):
        if i >= int(num_classes):
            break
        label_to_class[int(lab)] = int(i)

    base = int(k + 1)
    bos = int(k)

    tpls = list(templates)
    w_ctx = float(sum(float(t.weight) for t in tpls))
    w_uni = float(unigram_weight)
    if w_ctx < 0.0 or w_uni < 0.0:
        raise ValueError("weights must be non-negative")
    if (w_ctx + w_uni) <= 0.0:
        raise ValueError("At least one weight must be > 0")
    # Normalize weights.
    s = float(w_ctx + w_uni)
    tpls = [ContextTemplate(name=t.name, offsets=t.offsets, weight=float(t.weight) / s) for t in tpls]
    w_uni = float(w_uni) / s

    models: list[ClassCountsModel] = []
    for _ in range(int(num_classes)):
        models.append(
            ClassCountsModel(
                k=int(k),
                alpha=float(alpha),
                templates=list(tpls),
                unigram_weight=float(w_uni),
                base=int(base),
                uni=np.zeros((int(k),), dtype=np.int64),
                uni_total=0,
                tables=[{} for _ in tpls],
            )
        )

    n, ht, wt = grids.shape
    for idx in range(int(n)):
        lab = int(y[idx])
        ci = label_to_class.get(lab, None)
        if ci is None:
            continue
        m = models[int(ci)]
        grid = grids[idx]

        flat = grid.reshape(-1).astype(np.int64, copy=False)
        m.uni += np.bincount(flat, minlength=int(k)).astype(np.int64)

        for i in range(int(ht)):
            for j in range(int(wt)):
                tok = int(grid[i, j])
                for ti, tpl in enumerate(m.templates):
                    key = _ctx_key(grid, i=i, j=j, offsets=tpl.offsets, bos=bos, base=base)
                    _update_sparse(m.tables[ti], key, tok)

    for m in models:
        m.uni_total = int(m.uni.sum())
    return models, label_to_class


def loglik_grid(
    model: ClassCountsModel,
    grid: np.ndarray,
) -> float:
    """Compute log p(grid) under a class-conditional model (mixture of contexts + unigram)."""
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("grid must be (Ht,Wt)")
    k = int(model.k)
    base = int(model.base)
    bos = int(k)
    alpha = float(model.alpha)
    alpha_k = float(alpha * float(k))

    uni = model.uni
    denom_uni = float(model.uni_total + alpha_k)
    if denom_uni <= 0.0:
        denom_uni = float(alpha_k)

    ll = 0.0
    ht, wt = int(g.shape[0]), int(g.shape[1])
    for i in range(int(ht)):
        for j in range(int(wt)):
            tok = int(g[i, j])
            p = 0.0
            # Context mixture.
            for tpl, table in zip(model.templates, model.tables, strict=True):
                key = _ctx_key(g, i=i, j=j, offsets=tpl.offsets, bos=bos, base=base)
                p += float(tpl.weight) * _prob_sparse(
                    table.get(int(key)), tok, alpha=alpha, alpha_k=alpha_k, k=k
                )
            # Unigram backoff.
            if float(model.unigram_weight) > 0.0:
                p_uni = float((int(uni[tok]) + alpha) / denom_uni)
                p += float(model.unigram_weight) * float(p_uni)
            if p <= 0.0:
                p = 1e-12
            ll += float(np.log(float(p)))
    return float(ll)


def predict_class(
    models: Sequence[ClassCountsModel],
    grid: np.ndarray,
) -> int:
    scores = [loglik_grid(m, grid) for m in models]
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def sample_grid(
    model: ClassCountsModel,
    *,
    ht: int,
    wt: int,
    seed: int,
) -> np.ndarray:
    """Autoregressively sample a token grid from a model."""
    rng = np.random.default_rng(int(seed))
    k = int(model.k)
    bos = int(k)
    base = int(model.base)
    alpha = float(model.alpha)
    alpha_k = float(alpha * float(k))

    # Precompute normalized weights.
    w_ctx = [float(t.weight) for t in model.templates]
    w_uni = float(model.unigram_weight)
    s = float(sum(w_ctx) + w_uni)
    if s <= 0.0:
        raise ValueError("Invalid weights")
    w_ctx = [w / s for w in w_ctx]
    w_uni = w_uni / s

    def sample_from_sparse(ent: Optional[SparseCounts]) -> int:
        if ent is None or ent.total <= 0:
            return int(rng.integers(0, k))
        total = int(ent.total)
        if rng.random() < (float(total) / float(total + alpha_k)):
            r = int(rng.integers(0, total))
            s2 = 0
            for tok, cnt in ent.counts.items():
                s2 += int(cnt)
                if r < s2:
                    return int(tok)
            return int(next(iter(ent.counts.keys())))
        return int(rng.integers(0, k))

    def sample_from_unigram() -> int:
        if int(model.uni_total) <= 0:
            return int(rng.integers(0, k))
        if rng.random() < (float(model.uni_total) / float(model.uni_total + alpha_k)):
            probs = (model.uni / float(model.uni_total)).astype(np.float64, copy=False)
            return int(rng.choice(int(k), p=probs))
        return int(rng.integers(0, k))

    g = np.empty((int(ht), int(wt)), dtype=np.int32)
    for i in range(int(ht)):
        for j in range(int(wt)):
            r = float(rng.random())
            acc = 0.0
            chosen: int | None = None
            for tpl, table, w in zip(model.templates, model.tables, w_ctx, strict=True):
                acc += float(w)
                if r < acc:
                    key = _ctx_key(g, i=i, j=j, offsets=tpl.offsets, bos=bos, base=base)
                    chosen = sample_from_sparse(table.get(int(key)))
                    break
            if chosen is None:
                chosen = sample_from_unigram() if w_uni > 0.0 else int(rng.integers(0, k))
            g[i, j] = int(chosen)
    return g

