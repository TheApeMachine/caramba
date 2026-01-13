"""Decoupled KV cache for DBA attention.

DBA (Decoupled Bottleneck Attention) uses separate projections for semantic
and geometric keys, plus a separate value projection. This cache stores
three tensors instead of two: k_sem, k_geo, and v.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from caramba.cache.tensor import SeqCacheTensor

if TYPE_CHECKING:
    from collections.abc import Mapping
    from caramba.config.kvcache import KVCacheTensorConfig


class DecoupledLayerKVCache:
    """Stores k_sem, k_geo, and v caches for one DBA layer.

    Unlike standard attention which caches K and V of size n_kv_heads × head_dim,
    DBA caches:
    - k_sem: semantic keys (sem_dim)
    - k_geo: geometric keys (geo_dim)
    - v: values (v_dim)

    This is where the memory savings come from—these dimensions are typically
    much smaller than the full kv_dim.
    """

    k_sem: SeqCacheTensor
    k_geo: SeqCacheTensor
    v: SeqCacheTensor

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_sem_dim: int,
        k_geo_dim: int,
        v_dim: int,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ) -> None:
        """Allocate storage for all three cache tensors."""
        self.k_sem = SeqCacheTensor(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dim=int(k_sem_dim),
            cfg=k_sem_cfg,
            device=device,
        )
        self.k_geo = SeqCacheTensor(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dim=int(k_geo_dim),
            cfg=k_geo_cfg,
            device=device,
        )
        self.v = SeqCacheTensor(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dim=int(v_dim),
            cfg=v_cfg,
            device=device,
        )

    @property
    def pos(self) -> int:
        """Current sequence position (how many tokens are cached)."""
        return int(self.k_sem.pos)

    def append(
        self,
        k_sem_new: torch.Tensor,
        k_geo_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> int:
        """Append new tokens to all three caches.

        Returns the position before the append (where new tokens start).
        """
        k_pos = self.k_sem.append(k_sem_new)
        g_pos = self.k_geo.append(k_geo_new)
        v_pos = self.v.append(v_new)
        if int(k_pos) != int(g_pos) or int(k_pos) != int(v_pos):
            raise RuntimeError("Decoupled K/V append position mismatch")
        return int(k_pos)

    def get(
        self,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve all cached tokens as (k_sem, k_geo, v)."""
        return (
            self.k_sem.get(dtype=dtype),
            self.k_geo.get(dtype=dtype),
            self.v.get(dtype=dtype),
        )

    def get_slice(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a slice of cached tokens as (k_sem, k_geo, v)."""

        return (
            self.k_sem.get_slice(start, end, dtype=dtype),
            self.k_geo.get_slice(start, end, dtype=dtype),
            self.v.get_slice(start, end, dtype=dtype),
        )

    def truncate(self, new_pos: int) -> None:
        """Rollback all caches to a previous position."""
        self.k_sem.truncate(new_pos)
        self.k_geo.truncate(new_pos)
        self.v.truncate(new_pos)
        if not (self.k_sem.pos == self.k_geo.pos == self.v.pos):
            raise RuntimeError("Decoupled cache desync after truncate")

    @property
    def keys(self) -> tuple[str, ...]:
        """Stable field ordering for generic cache ops."""

        return ("k_sem", "k_geo", "v")

    def append_many(self, items: "Mapping[str, torch.Tensor]") -> int:
        """Generic append API for graph-composed ops."""

        want = {"k_sem", "k_geo", "v"}
        if set(items.keys()) != want:
            extra = sorted(set(items.keys()) - want)
            missing = sorted(want - set(items.keys()))
            raise KeyError(f"DecoupledLayerKVCache.append_many mismatch (missing={missing}, extra={extra})")
        return self.append(items["k_sem"], items["k_geo"], items["v"])

    def get_many(self, *, dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
        """Generic get API for graph-composed ops."""

        k_sem, k_geo, v = self.get(dtype=dtype)
        return {"k_sem": k_sem, "k_geo": k_geo, "v": v}

    def get_slice_many(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        """Generic slice API for graph-composed ops."""

        k_sem, k_geo, v = self.get_slice(start, end, dtype=dtype)
        return {"k_sem": k_sem, "k_geo": k_geo, "v": v}
