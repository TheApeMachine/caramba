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
from typing import Any, cast

import torch
from torch import Tensor, nn

from carmath import last_write_wins
from layer.memory_block.memory.vsa import VsaNovelty, VsaTagProjector
from layer.memory_block.state import MemoryBlockState


@dataclass(slots=True)
class MemoryWriter:
    """Memory writer

    Writes are a policy decision: when to store, where to store, and whether to
    overwrite a similar existing item or replace the least-recently used slot.
    """

    mem_wkey: nn.Linear
    mem_value: nn.Linear
    mem_write_gate: nn.Linear
    mem_buckets: int
    mem_table_buckets: int
    mem_hashes: int
    mem_assoc: int
    mem_key_dim: int
    mem_tag_dim: int
    mem_dim: int
    mem_write_threshold: float
    mem_write_eta: float
    mem_match_threshold: float
    mem_vsa_enabled: bool
    vsa_projector: VsaTagProjector | None
    vsa_novelty: VsaNovelty | None
    mem_trie_enabled: bool
    mem_trie_eta_decay: float
    mem_trie_max_levels: int | None
    tuner_mode: str = "off"
    _tuner_cache_mode: str | None = None
    _tuner_cache: Any | None = None

    def __post_init__(self) -> None:
        allowed_tuner_modes = {"off", "monitor", "adaptive"}
        if str(self.tuner_mode) not in allowed_tuner_modes:
            raise ValueError(
                f"Invalid tuner_mode={self.tuner_mode!r}; expected one of {sorted(allowed_tuner_modes)!r}"
            )
        if self.mem_vsa_enabled:
            if self.vsa_projector is None:
                raise ValueError("mem_vsa_enabled is True but vsa_projector is None")
            if self.vsa_novelty is None:
                raise ValueError("mem_vsa_enabled is True but vsa_novelty is None")

        if self.mem_trie_enabled:
            if int(self.mem_buckets) < 2:
                raise ValueError("mem_trie_enabled requires mem_buckets >= 2")
            if int(self.mem_buckets) & (int(self.mem_buckets) - 1) != 0:
                raise ValueError("mem_trie_enabled requires mem_buckets to be a power of two")
            expected_table_buckets = int(2 * int(self.mem_buckets) - 1)
            if int(self.mem_table_buckets) != expected_table_buckets:
                raise ValueError(
                    "mem_trie_enabled requires mem_table_buckets == 2 * mem_buckets - 1, "
                    f"got mem_table_buckets={int(self.mem_table_buckets)} mem_buckets={int(self.mem_buckets)}"
                )

    def _get_tuner(self) -> Any | None:
        mode = str(self.tuner_mode)
        if mode == "off":
            self._tuner_cache_mode = mode
            self._tuner_cache = None
            return None
        if self._tuner_cache is None or self._tuner_cache_mode != mode:
            from layer.memory_block.memory.tuner import get_shared_tuner

            self._tuner_cache = get_shared_tuner(mode=mode)
            self._tuner_cache_mode = mode
        return self._tuner_cache

    def write_chunk(
        self,
        u: Tensor,
        st: MemoryBlockState,
        routing: dict[str, Any],
        t0: int,
        mask: Tensor | None,
        write_scale: Tensor | None,
    ) -> Tensor:
        """Write a chunk of tokens

        Writes update runtime state (the memory tables), so they run under
        `no_grad` to avoid autograd tracking and in-place versioning issues.
        """
        B, T, idx_w, gate_logit, w_eta, do = self.prepare(u, routing, mask=mask, write_scale=write_scale)
        # Always expose "write_do" when collecting aux so tuner/telemetry can
        # see "no writes happened" and react.
        if bool(routing.get("collect_aux", False)):
            routing["write_do"] = do.detach()
            routing["write_eta"] = w_eta.detach()
        if not bool(do.any()):
            return gate_logit

        # Memory writes are runtime state updates, not part of the differentiable path.
        # Doing them under autograd causes in-place versioning errors (esp. on MPS).
        with torch.no_grad():
            # Clone mutable state ONCE per chunk write.
            # (Trie mode writes to multiple nodes; cloning per node explodes allocations.)
            self.ensure_mutable_state(st)

            wk = self.mem_wkey(u)
            if self.mem_vsa_enabled:
                wt = cast(VsaTagProjector, self.vsa_projector)(wk)
            else:
                wt = torch.zeros((B, T, int(self.mem_tag_dim)), device=u.device, dtype=wk.dtype)
            v = self.mem_value(u)
            pos = torch.nonzero(do, as_tuple=False)
            b_ev_all = pos[:, 0]
            t_ev_all = pos[:, 1]

            novelty_sum = torch.zeros((B, T), device=u.device, dtype=u.dtype)
            max_sim_sum = torch.zeros((B, T), device=u.device, dtype=u.dtype)
            for h in range(int(self.mem_hashes)):
                bidx_leaf = idx_w[:, :, h]
                if self.mem_trie_enabled:
                    bidx0 = self.trie_leaf_to_node(bidx_leaf)
                else:
                    bidx0 = bidx_leaf

                if self.mem_vsa_enabled:
                    novelty_h, max_sim_h = self.hash_novelty(st, bidx=bidx0, wt=wt, h=int(h))
                else:
                    novelty_h = torch.ones((B, T), device=u.device, dtype=u.dtype)
                    max_sim_h = torch.zeros((B, T), device=u.device, dtype=u.dtype)
                novelty_sum = novelty_sum + novelty_h
                max_sim_sum = max_sim_sum + max_sim_h
                if self.mem_trie_enabled:
                    self.write_trie(
                        u=u,
                        st=st,
                        bidx_leaf=bidx_leaf,
                        wk=wk,
                        wt=wt,
                        v=v,
                        b_ev_all=b_ev_all,
                        t_ev_all=t_ev_all,
                        w_eta=(w_eta * novelty_h),
                        h=int(h),
                        t0=int(t0),
                    )
                else:
                    self.write_hash(
                        u=u,
                        st=st,
                        bidx=bidx0,
                        wk=wk,
                        wt=wt,
                        v=v,
                        b_ev_all=b_ev_all,
                        t_ev_all=t_ev_all,
                        w_eta=(w_eta * novelty_h),
                        h=int(h),
                        t0=int(t0),
                    )
            if bool(routing.get("collect_aux", False)):
                denom = float(max(1, int(self.mem_hashes)))
                routing["write_novelty"] = (novelty_sum / denom).detach()
                routing["write_max_sim_vsa"] = (max_sim_sum / denom).detach()
        return gate_logit

    def trie_leaf_to_node(self, bidx_leaf: Tensor) -> Tensor:
        """Map leaf bucket ids (0..L-1) to trie node indices (base..base+L-1)."""
        leaves = int(self.mem_buckets)
        base = int(leaves - 1)
        return bidx_leaf.to(dtype=torch.long) + base

    def write_trie(
        self,
        *,
        u: Tensor,
        st: MemoryBlockState,
        bidx_leaf: Tensor,
        wk: Tensor,
        wt: Tensor,
        v: Tensor,
        b_ev_all: Tensor,
        t_ev_all: Tensor,
        w_eta: Tensor,
        h: int,
        t0: int,
    ) -> None:
        """Write to leaf node and its ancestors with geometric eta decay."""
        leaves = int(self.mem_buckets)
        max_depth = int((leaves - 1).bit_length())
        if self.mem_trie_max_levels is not None:
            max_depth = min(max_depth, int(self.mem_trie_max_levels))
        cur = self.trie_leaf_to_node(bidx_leaf)
        eta = w_eta
        for _level in range(int(max_depth) + 1):
            self.write_hash(
                u=u,
                st=st,
                bidx=cur,
                wk=wk,
                wt=wt,
                v=v,
                b_ev_all=b_ev_all,
                t_ev_all=t_ev_all,
                w_eta=eta,
                h=int(h),
                t0=int(t0),
            )
            if int(_level) >= int(max_depth):
                break
            if not bool((cur > 0).any()):
                break
            cur = torch.where(cur > 0, (cur - 1) // 2, cur)
            eta = eta * float(self.mem_trie_eta_decay)

    def prepare(
        self,
        u: Tensor,
        routing: dict[str, Any],
        *,
        mask: Tensor | None,
        write_scale: Tensor | None,
    ) -> tuple[int, int, Tensor, Tensor, Tensor, Tensor]:
        if u.ndim != 3:
            raise ValueError(f"u must have shape (B,T,D), got {tuple(u.shape)}")
        B, T, _ = u.shape
        idx_w = routing["idx_w"]
        gate_logit = self.mem_write_gate(u).squeeze(-1)
        p = torch.sigmoid(gate_logit)

        # Apply tuner scaling to thresholds
        write_threshold = float(self.mem_write_threshold)
        tuner = self._get_tuner()
        if tuner is not None:
            write_threshold = write_threshold * getattr(tuner, "write_threshold_mult", 1.0)

        m = (p > write_threshold).to(dtype=u.dtype)
        if mask is not None:
            m = torch.where(mask >= 0, (mask > 0).to(dtype=m.dtype, device=m.device), m)
        w_eta = (float(self.mem_write_eta) * p).to(dtype=u.dtype) * m
        if write_scale is not None:
            if write_scale.shape != (B, T):
                raise ValueError(f"write_scale must have shape (B,T)={(B,T)}, got {tuple(write_scale.shape)}")
            w_eta = w_eta * write_scale.to(dtype=u.dtype, device=u.device)
        do = w_eta > 0
        if bool(routing.get("collect_aux", False)):
            # Debuggable gating signals
            routing["write_gate_p"] = p.detach()
            routing["write_threshold_eff"] = torch.tensor(write_threshold, device=u.device, dtype=u.dtype)
        return int(B), int(T), idx_w, gate_logit, w_eta, do

    def hash_novelty(self, st: MemoryBlockState, *, bidx: Tensor, wt: Tensor, h: int) -> tuple[Tensor, Tensor]:
        if self.vsa_novelty is None:
            raise RuntimeError("mem_vsa_enabled is True but vsa_novelty is None")
        if int(h) < 0 or int(h) >= int(self.mem_hashes):
            raise ValueError(f"h must be in [0,{int(self.mem_hashes)-1}], got {int(h)}")
        B, T = int(bidx.size(0)), int(bidx.size(1))
        idx = bidx.to(dtype=torch.long).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        mt = st.mem_tag[:, int(h) : int(h) + 1, :, :, :]
        ml = st.mem_last[:, int(h) : int(h) + 1, :, :]
        bt = mt.gather(dim=2, index=idx.expand(B, 1, T, self.mem_assoc, self.mem_tag_dim)).squeeze(1)
        bl = ml.gather(dim=2, index=idx[..., 0].expand(B, 1, T, self.mem_assoc)).squeeze(1)
        valid = bl >= 0
        sim = (bt * wt.unsqueeze(2)).sum(dim=-1) * (1.0 / math.sqrt(float(self.mem_tag_dim)))
        sim = sim.masked_fill(~valid, float("-inf"))
        any_valid = valid.any(dim=-1)
        max_sim = sim.max(dim=-1).values
        max_sim = torch.where(any_valid, max_sim, torch.full_like(max_sim, float("-inf")))

        # Apply tuner scaling to novelty threshold
        novelty_threshold = None
        tuner = self._get_tuner()
        if tuner is not None and hasattr(self.vsa_novelty, "threshold"):
            novelty_threshold = float(self.vsa_novelty.threshold) * getattr(tuner, "vsa_novelty_mult", 1.0)

        novelty = self.vsa_novelty(max_sim, threshold=novelty_threshold)
        return novelty.to(dtype=wt.dtype), max_sim.to(dtype=wt.dtype)

    def write_hash(
        self,
        *,
        u: Tensor,
        st: MemoryBlockState,
        bidx: Tensor,
        wk: Tensor,
        wt: Tensor,
        v: Tensor,
        b_ev_all: Tensor,
        t_ev_all: Tensor,
        w_eta: Tensor,
        h: int,
        t0: int,
    ) -> None:
        B, T = int(u.size(0)), int(u.size(1))
        mk = st.mem_k[:, h, :, :, :]
        ml = st.mem_last[:, h, :, :]
        idxk = bidx.to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1).expand(B, T, self.mem_assoc, self.mem_key_dim)
        idxl = bidx.to(dtype=torch.long).unsqueeze(-1).expand(B, T, self.mem_assoc)
        bk_w = mk.gather(dim=1, index=idxk)
        bl_w = ml.gather(dim=1, index=idxl)
        valid_w = bl_w >= 0
        sim_w = (bk_w * wk.unsqueeze(2)).sum(dim=-1) * (1.0 / math.sqrt(float(self.mem_key_dim)))
        sim_w = sim_w.masked_fill(~valid_w, float("-inf"))
        slot, use_update = self.choose_slot(sim_w=sim_w, bl_w=bl_w, valid_w=valid_w)
        self.apply_events(
            st=st,
            bidx=bidx,
            wk=wk,
            wt=wt,
            v=v,
            w_eta=w_eta,
            b_ev_all=b_ev_all,
            t_ev_all=t_ev_all,
            slot=slot,
            use_update=use_update,
            h=int(h),
            t0=int(t0),
        )

    def choose_slot(self, *, sim_w: Tensor, bl_w: Tensor, valid_w: Tensor) -> tuple[Tensor, Tensor]:
        best_slot = sim_w.argmax(dim=-1)
        best_sim = sim_w.max(dim=-1).values
        has_empty = (~valid_w).any(dim=-1)
        first_empty = (~valid_w).to(torch.int64).argmax(dim=-1)
        lru_slot = bl_w.argmin(dim=-1)
        repl_slot = torch.where(has_empty, first_empty, lru_slot)
        use_update = torch.isfinite(best_sim) & (best_sim >= float(self.mem_match_threshold)) & has_empty.logical_not()
        slot = torch.where(use_update, best_slot, repl_slot).to(dtype=torch.long)
        return slot, use_update

    def apply_events(
        self,
        *,
        st: MemoryBlockState,
        bidx: Tensor,
        wk: Tensor,
        wt: Tensor,
        v: Tensor,
        w_eta: Tensor,
        b_ev_all: Tensor,
        t_ev_all: Tensor,
        slot: Tensor,
        use_update: Tensor,
        h: int,
        t0: int,
    ) -> None:
        pack = self.select_events(
            st=st,
            bidx=bidx,
            wk=wk,
            wt=wt,
            v=v,
            w_eta=w_eta,
            b_ev_all=b_ev_all,
            t_ev_all=t_ev_all,
            slot=slot,
            use_update=use_update,
            t0=int(t0),
        )
        b_ev, bucket_ev, slot_ev, eta_ev, wk_ev, wt_ev, v_ev, upd_ev, time_ev = self.order_events(
            b_ev_all=b_ev_all,
            pack=pack,
            h=int(h),
        )
        if b_ev is None:
            return
        self.apply_updates(
            st=st,
            b_ev=b_ev,
            bucket_ev=bucket_ev,
            slot_ev=slot_ev,
            eta_ev=eta_ev,
            wk_ev=wk_ev,
            wt_ev=wt_ev,
            v_ev=v_ev,
            upd_ev=upd_ev,
            time_ev=time_ev,
            h=int(h),
        )

    def select_events(
        self,
        *,
        st: MemoryBlockState,
        bidx: Tensor,
        wk: Tensor,
        wt: Tensor,
        v: Tensor,
        w_eta: Tensor,
        b_ev_all: Tensor,
        t_ev_all: Tensor,
        slot: Tensor,
        use_update: Tensor,
        t0: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bucket_ev = bidx[b_ev_all, t_ev_all].to(dtype=torch.long)
        slot_ev = slot[b_ev_all, t_ev_all].to(dtype=torch.long)
        eta_ev = w_eta[b_ev_all, t_ev_all].to(dtype=wk.dtype)
        wk_ev = wk[b_ev_all, t_ev_all, :].to(dtype=wk.dtype)
        wt_ev = wt[b_ev_all, t_ev_all, :].to(dtype=wk.dtype)
        v_ev = v[b_ev_all, t_ev_all, :].to(dtype=wk.dtype)
        upd_ev = use_update[b_ev_all, t_ev_all]
        time_ev = (int(st.step) + int(t0)) + t_ev_all.to(torch.long)
        return bucket_ev, slot_ev, eta_ev, wk_ev, wt_ev, v_ev, upd_ev, time_ev

    def order_events(
        self,
        *,
        b_ev_all: Tensor,
        pack: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        h: int,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bucket_ev, slot_ev, eta_ev, wk_ev, wt_ev, v_ev, upd_ev, time_ev = pack
        buckets = int(self.mem_table_buckets) if int(self.mem_table_buckets) > 0 else int(self.mem_buckets)
        global_key = ((b_ev_all * int(self.mem_hashes) + int(h)) * buckets + bucket_ev) * int(self.mem_assoc) + slot_ev
        order = last_write_wins(global_key, time_ev)
        if order.numel() <= 0:
            f = torch.empty((0,), device=b_ev_all.device, dtype=eta_ev.dtype)
            m = torch.empty((0, self.mem_key_dim), device=b_ev_all.device, dtype=wk_ev.dtype)
            t = torch.empty((0, self.mem_tag_dim), device=b_ev_all.device, dtype=wt_ev.dtype)
            v0 = torch.empty((0, self.mem_dim), device=b_ev_all.device, dtype=v_ev.dtype)
            b0 = torch.empty((0,), device=b_ev_all.device, dtype=torch.long)
            return None, b0, b0, f, m, t, v0, (b0 > 0), b0
        sel = order
        return (
            b_ev_all[sel],
            bucket_ev[sel],
            slot_ev[sel],
            eta_ev[sel],
            wk_ev[sel],
            wt_ev[sel],
            v_ev[sel],
            upd_ev[sel],
            time_ev[sel],
        )

    def ensure_mutable_state(self, st: MemoryBlockState) -> None:
        st.mem_k = st.mem_k.clone()
        st.mem_v = st.mem_v.clone()
        st.mem_tag = st.mem_tag.clone()
        st.mem_last = st.mem_last.clone()

    def apply_updates(
        self,
        *,
        st: MemoryBlockState,
        b_ev: Tensor,
        bucket_ev: Tensor,
        slot_ev: Tensor,
        eta_ev: Tensor,
        wk_ev: Tensor,
        wt_ev: Tensor,
        v_ev: Tensor,
        upd_ev: Tensor,
        time_ev: Tensor,
        h: int,
    ) -> None:
        curv = st.mem_v[b_ev, int(h), bucket_ev, slot_ev, :]
        eta_view = eta_ev.view(-1, 1)
        newv = torch.where(upd_ev.view(-1, 1), (1.0 - eta_view) * curv + eta_view * v_ev, v_ev)
        st.mem_k[b_ev, int(h), bucket_ev, slot_ev, :] = wk_ev
        st.mem_v[b_ev, int(h), bucket_ev, slot_ev, :] = newv
        st.mem_tag[b_ev, int(h), bucket_ev, slot_ev, :] = wt_ev
        st.mem_last[b_ev, int(h), bucket_ev, slot_ev] = time_ev.to(dtype=torch.long)

