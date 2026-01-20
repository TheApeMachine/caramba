"""Small statistical helpers for benchmark reporting.

Focused on paired comparisons (same test cases evaluated by two models),
so small deltas like 0.88 vs 0.93 become interpretable via confidence
intervals and paired tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WilsonCI:
    low: float
    high: float


def wilson_ci(k: int, n: int, *, z: float = 1.959963984540054) -> WilsonCI:
    """Wilson score CI for a binomial proportion."""
    n = int(n)
    k = int(k)
    if n <= 0:
        return WilsonCI(0.0, 0.0)
    p = float(k) / float(n)
    zz = float(z) * float(z)
    denom = 1.0 + zz / float(n)
    center = (p + zz / (2.0 * float(n))) / denom
    rad = (float(z) * math.sqrt((p * (1.0 - p) + zz / (4.0 * float(n))) / float(n))) / denom
    return WilsonCI(low=max(0.0, center - rad), high=min(1.0, center + rad))


@dataclass(frozen=True, slots=True)
class PairedDeltaCI:
    delta: float
    low: float
    high: float


def paired_bootstrap_delta_ci(
    a: list[float],
    b: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> PairedDeltaCI:
    """Paired bootstrap CI for mean(a - b), sampling indices with replacement."""
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_delta_ci requires equal-length lists")
    n = len(a)
    if n == 0:
        return PairedDeltaCI(delta=0.0, low=0.0, high=0.0)
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    delta = sum(diffs) / float(n)

    rng = random.Random(int(seed))
    boots: list[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        for _j in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / float(n))
    boots.sort()

    lo_idx = int((alpha / 2.0) * len(boots))
    hi_idx = int((1.0 - alpha / 2.0) * len(boots)) - 1
    lo = boots[max(0, min(len(boots) - 1, lo_idx))]
    hi = boots[max(0, min(len(boots) - 1, hi_idx))]
    return PairedDeltaCI(delta=float(delta), low=float(lo), high=float(hi))


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Exact McNemar test p-value (two-sided) using binomial distribution.

    b = # cases where A correct, B wrong
    c = # cases where A wrong, B correct
    """
    b = int(b)
    c = int(c)
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)

    # Compute P(X <= k) for X ~ Binomial(n, 0.5)
    # two-sided p = 2 * min(P(X<=k), P(X>=n-k))
    # symmetry -> P(X>=n-k) == P(X<=k)
    p_le = 0.0
    for i in range(0, k + 1):
        p_le += math.comb(n, i) * (0.5 ** n)
    p = 2.0 * p_le
    return float(min(1.0, max(0.0, p)))

