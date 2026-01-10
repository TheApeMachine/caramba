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

from caramba.layer.memory_block.memory.vsa import VsaTagProjector
from caramba.layer.memory_block.memory.phase import PhaseSimilarity, PhaseTagProjector
from caramba.layer.memory_block.state import MemoryBlockState


@dataclass(slots=True)
class MemoryReader:
    """Memory reader

    Reads are “content addressed within a bucket”: routing picks a small set of
    candidate slots, then similarity scoring decides which slot(s) to use.
    """

    mem_out: nn.Linear
    mem_qkey: nn.Linear
    mem_key_dim: int
    mem_dim: int
    mem_tag_dim: int
    mem_assoc: int
    mem_hashes: int
    mem_buckets: int
    # `mem_buckets` is the number of leaf buckets (routing space). `mem_table_buckets`
    # is the physical table size (state space): in trie mode it includes internal nodes
    # (e.g., `mem_table_buckets = 2 * mem_buckets - 1`), otherwise it equals `mem_buckets`.
    mem_table_buckets: int
    mem_read_temp: float
    mem_vsa_weight: float
    mem_vsa_enabled: bool
    vsa_projector: VsaTagProjector | None
    mem_phase_weight: float
    mem_phase_enabled: bool
    phase_projector: PhaseTagProjector | None
    phase_similarity: PhaseSimilarity | None
    mem_trie_enabled: bool
    mem_trie_fallback_enabled: bool
    mem_trie_max_levels: int | None = None
    tuner: Any | None = None

    def read(self, u: Tensor, st: MemoryBlockState, routing: dict[str, Any]) -> Tensor:
        """Read memory for a chunk

        The returned vector is a learned projection of weighted slot values,
        which makes the memory interface compatible with the model dimension.
        """
        B, T = self.validate(u)
        idx_g, valid_hint = self.gather_index_and_valid(st, routing, batch=B, time=T)
        bk, bv, bt, valid = self.gather_bucket(st, idx_g=idx_g, batch=B, time=T)
        if valid_hint is not None:
            valid = valid_hint
        qk = self.mem_qkey(u)
        sim_key = self.score_key(bk=bk, qk=qk, valid=valid, batch=B, time=T)
        if self.mem_vsa_enabled and float(self.mem_vsa_weight) != 0.0:
            if self.vsa_projector is None:
                raise RuntimeError("mem_vsa_enabled is True but vsa_projector is None")
            qt = self.vsa_projector(qk)
            sim_vsa = self.score_vsa(bt=bt, qt=qt, valid=valid, batch=B, time=T)
            
            # Apply tuner scaling to VSA weight
            vsa_weight = float(self.mem_vsa_weight)
            if self.tuner is not None:
                vsa_weight = vsa_weight * getattr(self.tuner, "vsa_novelty_mult", 1.0)
                
            sim_total = sim_key + vsa_weight * sim_vsa
        else:
            sim_vsa = torch.zeros_like(sim_key)
            sim_total = sim_key
        if self.mem_phase_enabled and float(self.mem_phase_weight) != 0.0:
            if self.phase_projector is None or self.phase_similarity is None:
                raise RuntimeError("mem_phase_enabled is True but phase modules are missing")
            qphi = self.phase_projector(qk)
            kphi = self.phase_projector(bk)
            sim_phase = self.phase_similarity.score(q_angles=qphi, k_angles=kphi, valid=valid, batch=B, time=T)
            sim_total = sim_total + float(self.mem_phase_weight) * sim_phase
        else:
            sim_phase = torch.zeros_like(sim_key)
        w = self.slot_weights(sim=sim_total, valid=valid)
        read_h = (w.unsqueeze(-1) * bv).sum(dim=3)
        if bool(routing.get("collect_aux", False)):
            routing["read_slot_weights"] = w.detach()
            routing["read_slot_sim"] = sim_total.detach()
            routing["read_slot_sim_vsa"] = sim_vsa.detach()
            routing["read_slot_sim_phase"] = sim_phase.detach()
        return self.mem_out(read_h.sum(dim=1))

    def validate(self, u: Tensor) -> tuple[int, int]:
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, _ = u.shape
        return int(B), int(T)

    def gather_index_and_valid(self, st: MemoryBlockState, routing: dict[str, Any], *, batch: int, time: int) -> tuple[Tensor, Tensor | None]:
        idx_r = routing["idx_r"]
        if idx_r.ndim != 3:
            raise ValueError(f"routing['idx_r'] must have shape (B,T,H), got {tuple(idx_r.shape)}")
        idx = idx_r.permute(0, 2, 1).to(dtype=torch.long)  # (B,H,T) in leaf bucket space
        valid_hint: Tensor | None = None
        if self.mem_trie_enabled:
            leaves = int(self.mem_buckets)
            base = int(leaves - 1)
            idx = idx + base  # leaf node indices in [base, base+leaves-1]
            if self.mem_trie_fallback_enabled:
                idx, valid_hint, steps = self.trie_fallback_index(st, idx=idx, batch=batch, time=time)
                if bool(routing.get("collect_aux", False)) and isinstance(steps, Tensor):
                    routing["trie_fallback_steps"] = steps.detach()
        return idx.unsqueeze(-1).unsqueeze(-1), valid_hint

    def trie_fallback_index(self, st: MemoryBlockState, *, idx: Tensor, batch: int, time: int) -> tuple[Tensor, Tensor, Tensor]:
        """Fallback up trie parents when leaf bucket is empty.

        Args:
            idx: (B,H,T) node indices in trie table (0..mem_table_buckets-1)

        Returns:
            (idx_final, valid_final, steps) where:
            - idx_final: (B,H,T) final node index after fallback
            - valid_final: (B,H,T,A) validity mask for that node's slots. Slots are
              considered valid when `mem_last` is non-negative (`bl >= 0`).
            - steps: (B,H,T) number of parent steps taken

        Notes:
            - The root is a fixed point: when `cur == 0`, `parent` stays 0 via `torch.where`,
              so fallback cannot walk past the root.
        """
        if int(self.mem_buckets) < 2:
            raise ValueError("Trie requires mem_buckets (leaf count) >= 2")
        max_depth = int((int(self.mem_buckets) - 1).bit_length())
        if self.mem_trie_max_levels is not None:
            max_depth = min(max_depth, int(self.mem_trie_max_levels))
        steps = torch.zeros((batch, self.mem_hashes, time), device=idx.device, dtype=torch.long)
        cur = idx
        valid = torch.zeros((batch, self.mem_hashes, time, self.mem_assoc), device=idx.device, dtype=torch.bool)
        for _ in range(int(max_depth) + 1):
            idxl = cur.unsqueeze(-1).expand(batch, self.mem_hashes, time, self.mem_assoc)
            bl = st.mem_last.gather(dim=2, index=idxl)
            valid = bl >= 0
            any_valid = valid.any(dim=-1)  # (B,H,T)
            need = ~any_valid
            if not bool(need.any()):
                break
            parent = torch.where(cur > 0, (cur - 1) // 2, cur)
            cur = torch.where(need, parent, cur)
            steps = steps + need.to(dtype=torch.long)
        return cur, valid, steps

    def gather_bucket(self, st: MemoryBlockState, *, idx_g: Tensor, batch: int, time: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        
        # Lever for future expansion: read_temp_mult
        read_temp = float(self.mem_read_temp)
        if self.tuner is not None:
            read_temp = read_temp * getattr(self.tuner, "read_temp_mult", 1.0)
            
        w = torch.softmax(sim / float(max(1e-6, read_temp)), dim=-1)
        return torch.where(any_valid, w, torch.zeros_like(w))

