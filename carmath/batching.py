"""Batch sizing helpers."""

from __future__ import annotations


def token_budget_batch_size(
    base_batch_size: int,
    *,
    block_size: int,
    ref_block_size: int,
    min_batch_size: int = 1,
) -> int:
    """Scale batch size assuming a roughly constant tokens-per-batch budget.

    Heuristic: batch_size ∝ ref_block_size / block_size.
    """
    bs = int(base_batch_size)
    ref = int(ref_block_size)
    bsz = int(block_size)
    min_bs = int(min_batch_size)
    if bs <= 0:
        bs = 1
    if ref <= 0 or bsz <= 0:
        return max(min_bs, bs)
    scaled = int(bs * (ref / float(bsz)))
    return max(min_bs, int(scaled))

