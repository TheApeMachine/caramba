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

from caramba.layer.mosaic.memory.vsa import VsaTagProjector
from caramba.layer.mosaic.state import MosaicState


@dataclass(slots=True)
class MemoryReader:
    """Memory reader."""

    mem_out: nn.Linear
    mem_qkey: nn.Linear
    mem_key_dim: int
    mem_dim: int
    mem_tag_dim: int
    mem_assoc: int
    mem_hashes: int
    mem_buckets: int
    mem_read_temp: float
    mem_vsa_weight: float
    mem_vsa_enabled: bool
    vsa_projector: VsaTagProjector | None

    def read(self, u: Tensor, st: MosaicState, routing: dict[str, Any]) -> Tensor:
        """Read memory for a chunk."""
        B, T = self.validate(u)
        idx_g = self.gather_index(routing)
        bk, bv, bt, valid = self.gather_bucket(st, idx_g=idx_g, batch=B, time=T)
        qk = self.mem_qkey(u)
        sim_key = self.score_key(bk=bk, qk=qk, valid=valid, batch=B, time=T)
        if self.mem_vsa_enabled and float(self.mem_vsa_weight) != 0.0:
            if self.vsa_projector is None:
                raise RuntimeError("mem_vsa_enabled is True but vsa_projector is None")
            qt = self.vsa_projector(qk)
            sim_vsa = self.score_vsa(bt=bt, qt=qt, valid=valid, batch=B, time=T)
            sim_total = sim_key + float(self.mem_vsa_weight) * sim_vsa
        else:
            sim_vsa = torch.zeros_like(sim_key)
            sim_total = sim_key
        w = self.slot_weights(sim=sim_total, valid=valid)
        read_h = (w.unsqueeze(-1) * bv).sum(dim=3)
        if bool(routing.get("collect_aux", False)):
            routing["read_slot_weights"] = w.detach()
            routing["read_slot_sim"] = sim_total.detach()
            routing["read_slot_sim_vsa"] = sim_vsa.detach()
        return self.mem_out(read_h.sum(dim=1))

    def validate(self, u: Tensor) -> tuple[int, int]:
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, _ = u.shape
        return int(B), int(T)

    def gather_index(self, routing: dict[str, Any]) -> Tensor:
        idx_r = routing["idx_r"]
        return idx_r.permute(0, 2, 1).to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)

    def gather_bucket(self, st: MosaicState, *, idx_g: Tensor, batch: int, time: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bk = st.mem_k.gather(dim=2, index=idx_g.expand(batch, self.mem_hashes, time, self.mem_assoc, self.mem_key_dim))
        bv = st.mem_v.gather(dim=2, index=idx_g.expand(batch, self.mem_hashes, time, self.mem_assoc, self.mem_dim))
        bt = st.mem_tag.gather(dim=2, index=idx_g.expand(batch, self.mem_hashes, time, self.mem_assoc, self.mem_tag_dim))
        bl = st.mem_last.gather(dim=2, index=idx_g[..., 0].expand(batch, self.mem_hashes, time, self.mem_assoc))
        return bk, bv, bt, bl >= 0

    def score_key(self, *, bk: Tensor, qk: Tensor, valid: Tensor, batch: int, time: int) -> Tensor:
        sim = (bk * qk.view(batch, 1, time, 1, self.mem_key_dim)).sum(dim=-1)
        sim = sim * (1.0 / math.sqrt(float(self.mem_key_dim)))
        return sim.masked_fill(~valid, float("-inf"))

    def score_vsa(self, *, bt: Tensor, qt: Tensor, valid: Tensor, batch: int, time: int) -> Tensor:
        sim = (bt * qt.view(batch, 1, time, 1, self.mem_tag_dim)).sum(dim=-1)
        sim = sim * (1.0 / math.sqrt(float(self.mem_tag_dim)))
        return sim.masked_fill(~valid, float("-inf"))

    def slot_weights(self, *, sim: Tensor, valid: Tensor) -> Tensor:
        any_valid = valid.any(dim=-1, keepdim=True)
        w = torch.softmax(sim / float(max(1e-6, self.mem_read_temp)), dim=-1)
        return torch.where(any_valid, w, torch.zeros_like(w))

