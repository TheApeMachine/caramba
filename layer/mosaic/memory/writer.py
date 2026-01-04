"""Memory writer

Performs constant-time set-associative writes:
- compute tag/value projections
- decide write via gate + threshold
- choose slot via match-threshold or LRU
- apply last-write-wins conflict resolution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from caramba.carmath import last_write_wins
from caramba.layer.mosaic.state import MosaicState


@dataclass(slots=True)
class MemoryWriter:
    """Memory writer."""

    mem_wkey: nn.Linear
    mem_value: nn.Linear
    mem_write_gate: nn.Linear
    mem_buckets: int
    mem_hashes: int
    mem_assoc: int
    mem_key_dim: int
    mem_dim: int
    mem_write_threshold: float
    mem_write_eta: float
    mem_match_threshold: float

    def write_chunk(
        self,
        u: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        t0: int,
        mask: Tensor | None,
        write_scale: Tensor | None,
    ) -> Tensor:
        """Write a chunk and return gate logits (B,T)."""
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, _ = u.shape
        idx_w = routing["idx_w"]
        gate_logit = self.mem_write_gate(u).squeeze(-1)
        p = torch.sigmoid(gate_logit)
        thr = float(self.mem_write_threshold)
        eta = float(self.mem_write_eta)
        m = (p > thr).to(dtype=u.dtype)
        if mask is not None:
            m = torch.where(mask >= 0, (mask > 0).to(dtype=m.dtype, device=m.device), m)
        w_eta = (float(eta) * p).to(dtype=u.dtype) * m
        if write_scale is not None:
            if write_scale.shape != (B, T):
                raise ValueError(f"write_scale must have shape (B,T)={(B,T)}, got {tuple(write_scale.shape)}")
            w_eta = w_eta * write_scale.to(dtype=u.dtype, device=u.device)
        do = w_eta > 0
        if not bool(do.any()):
            return gate_logit
        if bool(routing.get("collect_aux", False)):
            routing["write_do"] = do.detach()

        wk = self.mem_wkey(u)
        v = self.mem_value(u)
        pos = torch.nonzero(do, as_tuple=False)
        b_ev_all = pos[:, 0]
        t_ev_all = pos[:, 1]

        for h in range(int(self.mem_hashes)):
            bidx = idx_w[:, :, h].to(dtype=torch.long)
            mk = st.mem_k[:, h, :, :, :]
            mv = st.mem_v[:, h, :, :, :]
            ml = st.mem_last[:, h, :, :]
            idxk = bidx.unsqueeze(-1).unsqueeze(-1).expand(B, T, self.mem_assoc, self.mem_key_dim)
            idxl = bidx.unsqueeze(-1).expand(B, T, self.mem_assoc)
            bk_w = mk.gather(dim=1, index=idxk)
            bl_w = ml.gather(dim=1, index=idxl)
            valid_w = bl_w >= 0
            sim_w = (bk_w * wk.unsqueeze(2)).sum(dim=-1) * (1.0 / math.sqrt(float(self.mem_key_dim)))
            sim_w = sim_w.masked_fill(~valid_w, float("-inf"))
            best_slot = sim_w.argmax(dim=-1)
            best_sim = sim_w.max(dim=-1).values
            has_empty = (~valid_w).any(dim=-1)
            first_empty = (~valid_w).to(torch.int64).argmax(dim=-1)
            lru_slot = bl_w.argmin(dim=-1)
            repl_slot = torch.where(has_empty, first_empty, lru_slot)
            use_update = torch.isfinite(best_sim) & (best_sim >= float(self.mem_match_threshold)) & has_empty.logical_not()
            slot = torch.where(use_update, best_slot, repl_slot).to(dtype=torch.long)

            bucket_ev = bidx[b_ev_all, t_ev_all]
            slot_ev = slot[b_ev_all, t_ev_all]
            eta_ev = w_eta[b_ev_all, t_ev_all].to(dtype=u.dtype)
            wk_ev = wk[b_ev_all, t_ev_all, :].to(dtype=u.dtype)
            v_ev = v[b_ev_all, t_ev_all, :].to(dtype=u.dtype)
            upd_ev = use_update[b_ev_all, t_ev_all]
            time_ev = (int(st.step) + int(t0)) + t_ev_all.to(torch.long)

            global_key = ((b_ev_all * int(self.mem_hashes) + h) * int(self.mem_buckets) + bucket_ev) * int(self.mem_assoc) + slot_ev
            order = last_write_wins(global_key, time_ev)
            if order.numel() <= 0:
                continue

            b_ev = b_ev_all[order]
            bucket_ev = bucket_ev[order]
            slot_ev = slot_ev[order]
            eta_ev = eta_ev[order]
            wk_ev = wk_ev[order]
            v_ev = v_ev[order]
            upd_ev = upd_ev[order]
            time_ev = time_ev[order]

            st.mem_k = st.mem_k.clone()
            st.mem_v = st.mem_v.clone()
            st.mem_last = st.mem_last.clone()

            curv = st.mem_v[b_ev, h, bucket_ev, slot_ev, :]
            newv = torch.where(upd_ev.view(-1, 1), (1.0 - eta_ev.view(-1, 1)) * curv + eta_ev.view(-1, 1) * v_ev, v_ev)
            st.mem_k[b_ev, h, bucket_ev, slot_ev, :] = wk_ev
            st.mem_v[b_ev, h, bucket_ev, slot_ev, :] = newv
            st.mem_last[b_ev, h, bucket_ev, slot_ev] = time_ev.to(dtype=torch.long)

        return gate_logit

