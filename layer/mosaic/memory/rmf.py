"""Resonant Memory Field (RMF) for MOSAIC

RMF is a control-plane module that maintains a successor-biased activation field
over discrete memory addresses. It updates after each memory access and biases
future routing by adding a learned delta in key/tag space before routing.

This module is intentionally lightweight: it operates in O(rmf_dim) per step and
does not scan the memory table.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ResonantMemoryField(nn.Module):
    """Resonant Memory Field (RMF).

    RMF maintains a per-batch complex field vector and updates it from discrete
    bucket indices via a fixed phasor codebook. It can then produce a delta in
    key/tag space to bias routing.
    """

    def __init__(self, *, buckets: int, rmf_dim: int, key_dim: int, eta: float) -> None:
        super().__init__()
        if int(buckets) < 2:
            raise ValueError("buckets must be >= 2")
        if int(rmf_dim) < 1:
            raise ValueError("rmf_dim must be >= 1")
        if int(key_dim) < 1:
            raise ValueError("key_dim must be >= 1")
        if float(eta) < 0.0 or float(eta) > 1.0:
            raise ValueError("eta must be in [0,1]")

        self.buckets = int(buckets)
        self.rmf_dim = int(rmf_dim)
        self.key_dim = int(key_dim)
        self.eta = float(eta)

        angles = torch.empty((self.buckets, self.rmf_dim), dtype=torch.float32)
        angles.uniform_(-math.pi, math.pi)
        code_re = torch.cos(angles)
        code_im = torch.sin(angles)
        code = torch.stack([code_re, code_im], dim=-1)
        self.register_buffer("bucket_code", code, persistent=True)
        self.bucket_code: Tensor

        self.bias = nn.Linear(2 * int(self.rmf_dim), int(self.key_dim), bias=False)

    def initial_field(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if int(batch) < 1:
            raise ValueError("batch must be >= 1")
        f = torch.zeros((int(batch), int(self.rmf_dim), 2), device=device, dtype=dtype)
        f[:, :, 0] = 1.0
        return f

    def update(self, *, field: Tensor, idx_r: Tensor) -> Tensor:
        """Update field from read bucket indices.

        idx_r is expected to be (B, T, H) or (B, 1, H). We update once per step
        using the mean phasor code over hashes.
        """
        if idx_r.ndim != 3:
            raise ValueError(f"idx_r must have shape (B,T,H), got {tuple(idx_r.shape)}")
        if field.ndim != 3 or int(field.size(-1)) != 2:
            raise ValueError(f"field must have shape (B,rmf_dim,2), got {tuple(field.shape)}")

        B = int(idx_r.size(0))
        H = int(idx_r.size(2))
        idx = idx_r[:, -1, :].to(dtype=torch.long).clamp(0, int(self.buckets) - 1)
        code = self.bucket_code.to(device=field.device, dtype=field.dtype).index_select(0, idx.reshape(-1))
        code = code.view(B, H, self.rmf_dim, 2).mean(dim=1)
        new_field = self.mix(field=field, drive=code)
        return self.normalize(new_field)

    def mix(self, *, field: Tensor, drive: Tensor) -> Tensor:
        if float(self.eta) <= 0.0:
            return field
        return (1.0 - float(self.eta)) * field + float(self.eta) * drive

    def normalize(self, field: Tensor) -> Tensor:
        re = field[:, :, 0]
        im = field[:, :, 1]
        n = torch.sqrt(re * re + im * im).clamp_min(1e-6)
        return field / n.unsqueeze(-1)

    def routing_delta(self, *, field: Tensor) -> Tensor:
        """Compute a (B, key_dim) delta for routing key/tag space."""
        if field.ndim != 3 or int(field.size(-1)) != 2:
            raise ValueError(f"field must have shape (B,rmf_dim,2), got {tuple(field.shape)}")
        flat = field.reshape(int(field.size(0)), 2 * int(self.rmf_dim))
        return self.bias(flat)

    def field_history(self, *, idx_r: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Compute RMF field history from bucket indices.

        Args:
            idx_r: (B,T) or (B,T,H) int bucket indices (may contain -1 for ignore).
        Returns:
            Field history (B,T,rmf_dim,2) in the requested dtype/device.
        """
        idx = self.as_bucket_seq(idx_r)
        B, T = int(idx.size(0)), int(idx.size(1))
        codes = self.bucket_code.to(device=device, dtype=dtype)
        f = self.initial_field(batch=B, device=device, dtype=dtype)
        hist: list[Tensor] = []
        for t in range(T):
            hist.append(f)
            drive = self.drive(codes=codes, idx_t=idx[:, t])
            f = self.normalize(self.mix(field=f, drive=drive))
        return torch.stack(hist, dim=1)

    def as_bucket_seq(self, idx_r: Tensor) -> Tensor:
        if idx_r.ndim == 2:
            idx = idx_r
        elif idx_r.ndim == 3:
            idx = idx_r[:, :, 0]
        else:
            raise ValueError(f"idx_r must be (B,T) or (B,T,H), got {tuple(idx_r.shape)}")
        idx = idx.to(dtype=torch.long)
        idx = torch.where(idx >= 0, idx, torch.zeros_like(idx))
        return idx.clamp(0, int(self.buckets) - 1)

    def drive(self, *, codes: Tensor, idx_t: Tensor) -> Tensor:
        code = codes.index_select(0, idx_t.reshape(-1))
        return code.view(int(idx_t.size(0)), int(self.rmf_dim), 2)

