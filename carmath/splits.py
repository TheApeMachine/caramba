"""Dataset split math (train/val/test counts)."""

from __future__ import annotations


def train_val_counts(total: int, val_frac: float, *, min_val: int = 1) -> tuple[int, int]:
    """Compute (n_train, n_val) for a dataset split.

    Behavior:
    - If val_frac <= 0: returns (total, 0)
    - If val_frac > 0 and total > 1: ensures at least `min_val` validation samples
    """
    n = int(total)
    vf = float(val_frac)
    if n <= 0 or vf <= 0.0:
        return max(0, n), 0

    n_val = int(n * vf)
    if n > 1 and n_val < int(min_val):
        n_val = int(min_val)
    n_val = min(n_val, max(0, n - 1))  # keep at least 1 training sample when possible
    n_train = max(1, n - n_val) if n > 0 else 0
    return int(n_train), int(n_val)

