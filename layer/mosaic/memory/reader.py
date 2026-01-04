"""Memory reader

Performs constant-time set-associative reads:
- gather bucket slots
- score tag similarity
- softmax over associativity slots
- return projected read vector
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from caramba.layer.mosaic.state import MosaicState


@dataclass(slots=True)
class MemoryReader:
    """Memory reader."""

    mem_out: nn.Linear
    mem_qkey: nn.Linear
    mem_key_dim: int
    mem_dim: int
    mem_assoc: int
    mem_hashes: int
    mem_buckets: int
    mem_read_temp: float

    def read(self, u: Tensor, st: MosaicState, routing: dict[str, Any]) -> Tensor:
        """Read memory for a chunk."""
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, _ = u.shape
        idx_r = routing["idx_r"]
        qk = self.mem_qkey(u)
        mem_k = st.mem_k
        mem_v = st.mem_v
        mem_last = st.mem_last
        idx_g = idx_r.permute(0, 2, 1).to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)
        bk = mem_k.gather(dim=2, index=idx_g.expand(B, self.mem_hashes, T, self.mem_assoc, self.mem_key_dim))
        bv = mem_v.gather(dim=2, index=idx_g.expand(B, self.mem_hashes, T, self.mem_assoc, self.mem_dim))
        bl = mem_last.gather(dim=2, index=idx_g[..., 0].expand(B, self.mem_hashes, T, self.mem_assoc))
        valid = bl >= 0
        sim = (bk * qk.view(B, 1, T, 1, self.mem_key_dim)).sum(dim=-1) * (1.0 / math.sqrt(float(self.mem_key_dim)))
        sim = sim.masked_fill(~valid, float("-inf"))
        any_valid = valid.any(dim=-1, keepdim=True)
        w = torch.softmax(sim / float(max(1e-6, self.mem_read_temp)), dim=-1)
        w = torch.where(any_valid, w, torch.zeros_like(w))
        read_h = (w.unsqueeze(-1) * bv).sum(dim=3)
        if bool(routing.get("collect_aux", False)):
            routing["read_slot_weights"] = w.detach()
            routing["read_slot_sim"] = sim.detach()
        return self.mem_out(read_h.sum(dim=1))

