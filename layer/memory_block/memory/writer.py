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

from caramba.carmath import last_write_wins
from caramba.layer.memory_block.memory.vsa import VsaNovelty, VsaTagProjector
from caramba.layer.memory_block.state import MemoryBlockState


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
            from caramba.layer.memory_block.memory.tuner import get_shared_tuner

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
        """Write a chunk of tokens.

        By default, writes update runtime state (the memory tables) under
        `torch.no_grad()` to avoid autograd tracking and in-place versioning
        issues (especially on MPS).

        When `routing["differentiable_writes"]=True`, we instead update tables
        using *out-of-place* ops so gradients can flow through write→read within
        the same forward pass (needed for Table 2 / associative recall).
        """
        B, T, idx_w, gate_logit, w_eta, do = self.prepare(u, routing, mask=mask, write_scale=write_scale)
        # Always expose "write_do" when collecting aux so tuner/telemetry can
        # see "no writes happened" and react.
        if bool(routing.get("collect_aux", False)):
            routing["write_do"] = do.detach()
            routing["write_eta"] = w_eta.detach()
        if not bool(do.any()):
            return gate_logit

        differentiable = bool(routing.get("differentiable_writes", False))

        # Pre-compute common variables used by both code paths
        wk = self.mem_wkey(u)
        if self.mem_vsa_enabled:
            wt = cast(VsaTagProjector, self.vsa_projector)(wk)
        else:
            wt = torch.zeros((B, T, int(self.mem_tag_dim)), device=u.device, dtype=wk.dtype)
        v = self.mem_value(u)
        pos = torch.nonzero(do, as_tuple=False)
        b_ev_all = pos[:, 0]
        t_ev_all = pos[:, 1]

        # Debug: ensure b_ev_all and t_ev_all are 1D
        if b_ev_all.dim() != 1 or t_ev_all.dim() != 1:
            raise ValueError(f"b_ev_all or t_ev_all is not 1D: b_ev_all.shape={b_ev_all.shape}, t_ev_all.shape={t_ev_all.shape}")

        def _do_write() -> tuple[Tensor, Tensor]:

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
                        differentiable=differentiable,
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
                        differentiable=differentiable,
                    )
            return novelty_sum, max_sim_sum

        if differentiable:
            # Differentiable mode: use lazy cloning + batched out-of-place updates
            novelty_sum, max_sim_sum = _do_write()
        else:
            # Runtime-state update mode: use original approach for now
            with torch.no_grad():
                # Clone mutable state ONCE per chunk write.
                # (Trie mode writes to multiple nodes; cloning per node explodes allocations.)
                self.ensure_mutable_state(st)
                novelty_sum, max_sim_sum = _do_write()

        if bool(routing.get("collect_aux", False)):
            denom = float(max(1, int(self.mem_hashes)))
            routing["write_novelty"] = (novelty_sum / denom).detach()
            routing["write_max_sim_vsa"] = (max_sim_sum / denom).detach()
        return gate_logit

    def _do_write_inplace(
        self,
        u: Tensor,
        st: MemoryBlockState,
        routing: dict[str, Any],
        b_ev_all: Tensor,
        t_ev_all: Tensor,
        t0: int,
    ) -> tuple[Tensor, Tensor]:
        """In-place write operations with minimal memory overhead."""
        B, T = int(u.size(0)), int(u.size(1))
        wk = self.mem_wkey(u)
        if self.mem_vsa_enabled:
            wt = cast(VsaTagProjector, self.vsa_projector)(wk)
        else:
            wt = torch.zeros((B, T, int(self.mem_tag_dim)), device=u.device, dtype=wk.dtype)
        v = self.mem_value(u)

        novelty_sum = torch.zeros((B, T), device=u.device, dtype=u.dtype)
        max_sim_sum = torch.zeros((B, T), device=u.device, dtype=u.dtype)

        # Process all hashes in a single batched operation
        for h in range(int(self.mem_hashes)):
            bidx_leaf = routing["idx_w"][:, :, h]
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

            # Vectorized in-place updates for all events in this hash
            self._apply_hash_updates_inplace(
                st=st,
                bidx=bidx0,
                wk=wk,
                wt=wt,
                v=v,
                novelty_h=novelty_h,
                b_ev_all=b_ev_all,
                t_ev_all=t_ev_all,
                h=int(h),
                t0=int(t0),
            )

        return novelty_sum, max_sim_sum

    def _apply_hash_updates_inplace(
        self,
        *,
        st: MemoryBlockState,
        bidx: Tensor,
        wk: Tensor,
        wt: Tensor,
        v: Tensor,
        novelty_h: Tensor,
        b_ev_all: Tensor,
        t_ev_all: Tensor,
        h: int,
        t0: int,
    ) -> None:
        """Apply batched in-place updates for a single hash with sparse operations."""
        # Filter events for this hash that actually need updates (novelty > 0)
        try:
            valid_events = (novelty_h[b_ev_all, t_ev_all] > 0)
        except IndexError as e:
            # Debug: print tensor shapes if indexing fails
            print(f"DEBUG: novelty_h.shape={novelty_h.shape}, b_ev_all.shape={b_ev_all.shape}, t_ev_all.shape={t_ev_all.shape}")
            print(f"DEBUG: b_ev_all={b_ev_all}, t_ev_all={t_ev_all}")
            raise e

        if not valid_events.any():
            return

        b_ev = b_ev_all[valid_events]
        t_ev = t_ev_all[valid_events]

        # Get bucket indices for these events
        try:
            if bidx.ndim == 2:
                bucket_ev = bidx[b_ev, t_ev].to(dtype=torch.long)
            else:  # bidx.ndim == 3, shape (B, T, K)
                bucket_ev = bidx[b_ev, t_ev, h].to(dtype=torch.long)
        except IndexError as e:
            print(f"DEBUG: bidx.shape={bidx.shape}, b_ev.shape={b_ev.shape}, t_ev.shape={t_ev.shape}, h={h}")
            raise e

        # Only compute slot selection for events that pass the novelty threshold
        slot_ev, use_update = self._choose_slots_batched(
            st=st, b_ev=b_ev, bucket_ev=bucket_ev, wk=wk, wt=wt, t_ev=t_ev, h=h
        )

        if len(b_ev) == 0:
            return

        # Compute update values only for valid events
        eta_ev = novelty_h[b_ev, t_ev] * self.mem_write_eta
        wk_ev = wk[b_ev, t_ev, :]
        wt_ev = wt[b_ev, t_ev, :]
        v_ev = v[b_ev, t_ev, :]
        time_ev = (int(st.step) + int(t0)) + t_ev.to(torch.long)

        # Apply sparse in-place updates - only update locations that actually change
        h_idx = int(h)

        # For update operations (use_update=True): blend old and new values
        update_mask = use_update.view(-1, 1).expand(-1, self.mem_dim)
        curv = st.mem_v[b_ev, h_idx, bucket_ev, slot_ev, :]
        newv = torch.where(update_mask, (1.0 - eta_ev.view(-1, 1)) * curv + eta_ev.view(-1, 1) * v_ev, v_ev)

        # Sparse updates: only modify memory locations that are actually being written to
        # This reduces memory bandwidth compared to full tensor operations
        st.mem_k[b_ev, h_idx, bucket_ev, slot_ev, :] = wk_ev
        st.mem_v[b_ev, h_idx, bucket_ev, slot_ev, :] = newv
        st.mem_tag[b_ev, h_idx, bucket_ev, slot_ev, :] = wt_ev
        st.mem_last[b_ev, h_idx, bucket_ev, slot_ev] = time_ev

    def _choose_slots_batched(
        self,
        *,
        st: MemoryBlockState,
        b_ev: Tensor,
        bucket_ev: Tensor,
        wk: Tensor,
        wt: Tensor,
        t_ev: Tensor,
        h: int,
    ) -> tuple[Tensor, Tensor]:
        """Choose slots for batched events using vectorized operations."""
        h_idx = int(h)
        # Gather current memory state for all events at once
        bk_w = st.mem_k[b_ev, h_idx, bucket_ev, :, :]  # (N_events, mem_assoc, mem_key_dim)
        bl_w = st.mem_last[b_ev, h_idx, bucket_ev, :]   # (N_events, mem_assoc)

        # Compute similarities for all events at once
        wk_ev = wk[b_ev, t_ev, :]  # (N_events, mem_key_dim)
        sim_w = torch.einsum('nd, nad -> na', wk_ev, bk_w) * (1.0 / math.sqrt(float(self.mem_key_dim)))
        sim_w = sim_w.masked_fill(bl_w < 0, float("-inf"))

        # Find best slots
        best_sim, best_slot = sim_w.max(dim=-1)
        has_empty = (bl_w < 0).any(dim=-1)
        first_empty = (bl_w < 0).to(torch.int64).argmax(dim=-1)
        lru_slot = bl_w.argmin(dim=-1)

        repl_slot = torch.where(has_empty, first_empty, lru_slot)
        use_update = (best_sim >= float(self.mem_match_threshold)) & has_empty
        slot_ev = torch.where(use_update, best_slot, repl_slot).to(dtype=torch.long)

        return slot_ev, use_update

    def apply_single_event(
        self,
        *,
        st: MemoryBlockState,
        b_ev: int,
        t_ev: int,
        bucket_idx: int,
        wk: Tensor,
        wt: Tensor,
        v: Tensor,
        w_eta: Tensor,
        slot: int,
        use_update: bool,
        h: int,
        t0: int,
        differentiable: bool,
    ) -> None:
        """Apply a single write event."""
        eta_ev = float(w_eta[b_ev, t_ev])
        wk_ev = wk[b_ev, t_ev, :]
        wt_ev = wt[b_ev, t_ev, :]
        v_ev = v[b_ev, t_ev, :]
        time_ev = int(st.step) + int(t0) + int(t_ev)

        # Apply the update
        h_idx = int(h)
        curv = st.mem_v[b_ev, h_idx, bucket_idx, slot, :]
        if use_update:
            newv = (1.0 - eta_ev) * curv + eta_ev * v_ev
        else:
            newv = v_ev

        if differentiable:
            # Out-of-place updates for gradient flow - convert indices to tensors
            idx_tuple = (torch.tensor(b_ev), torch.tensor(h_idx), torch.tensor(bucket_idx), torch.tensor(slot))
            st.mem_k = st.mem_k.index_put(idx_tuple, wk_ev)
            st.mem_v = st.mem_v.index_put(idx_tuple, newv)
            st.mem_tag = st.mem_tag.index_put(idx_tuple, wt_ev)
            with torch.no_grad():
                st.mem_last[b_ev, h_idx, bucket_idx, slot] = time_ev
        else:
            # In-place updates
            st.mem_k[b_ev, h_idx, bucket_idx, slot, :] = wk_ev
            st.mem_v[b_ev, h_idx, bucket_idx, slot, :] = newv
            st.mem_tag[b_ev, h_idx, bucket_idx, slot, :] = wt_ev
            st.mem_last[b_ev, h_idx, bucket_idx, slot] = time_ev

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
        differentiable: bool,
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
                differentiable=differentiable,
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
        differentiable: bool,
    ) -> None:
        # Process each write event individually (not batched)
        for i in range(len(b_ev_all)):
            b_ev = int(b_ev_all[i])
            t_ev = int(t_ev_all[i])

            # Get bucket index for this event
            bucket_idx = int(bidx[b_ev, t_ev])

            # Gather memory state for this specific bucket
            bk_w = st.mem_k[b_ev, h, bucket_idx, :, :]  # (mem_assoc, mem_key_dim)
            bl_w = st.mem_last[b_ev, h, bucket_idx, :]   # (mem_assoc,)
            wk_ev = wk[b_ev, t_ev, :]  # (mem_key_dim,)

            # Compute similarities
            sim_w = (bk_w * wk_ev).sum(dim=-1) * (1.0 / math.sqrt(float(self.mem_key_dim)))
            valid_w = bl_w >= 0
            sim_w = sim_w.masked_fill(~valid_w, float("-inf"))

            # Choose slot for this event
            slot, use_update = self.choose_slot(sim_w=sim_w, bl_w=bl_w, valid_w=valid_w)

            # Apply the write event
            self.apply_single_event(
                st=st,
                b_ev=b_ev,
                t_ev=t_ev,
                bucket_idx=bucket_idx,
                wk=wk,
                wt=wt,
                v=v,
                w_eta=w_eta,
                slot=int(slot),
                use_update=bool(use_update),
                h=int(h),
                t0=int(t0),
                differentiable=differentiable,
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
        differentiable: bool,
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
            differentiable=differentiable,
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
        slot_ev = slot.to(dtype=torch.long)  # slot is already indexed by event
        eta_ev = w_eta[b_ev_all, t_ev_all].to(dtype=wk.dtype)
        wk_ev = wk[b_ev_all, t_ev_all, :].to(dtype=wk.dtype)
        wt_ev = wt[b_ev_all, t_ev_all, :].to(dtype=wk.dtype)
        v_ev = v[b_ev_all, t_ev_all, :].to(dtype=wk.dtype)
        upd_ev = use_update  # use_update is already indexed by event
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
        differentiable: bool,
    ) -> None:
        h_idx = torch.full_like(b_ev, int(h), dtype=torch.long)
        curv = st.mem_v[b_ev, h_idx, bucket_ev, slot_ev, :]
        eta_view = eta_ev.view(-1, 1)
        newv = torch.where(upd_ev.view(-1, 1), (1.0 - eta_view) * curv + eta_view * v_ev, v_ev)

        if differentiable:
            # Out-of-place updates: keep autograd happy.
            st.mem_k = st.mem_k.index_put((b_ev, h_idx, bucket_ev, slot_ev), wk_ev)
            st.mem_v = st.mem_v.index_put((b_ev, h_idx, bucket_ev, slot_ev), newv)
            st.mem_tag = st.mem_tag.index_put((b_ev, h_idx, bucket_ev, slot_ev), wt_ev)
            # mem_last is an int64 runtime bookkeeping tensor; keep it out of autograd.
            with torch.no_grad():
                st.mem_last[b_ev, h_idx, bucket_ev, slot_ev] = time_ev.to(dtype=torch.long)
        else:
            st.mem_k[b_ev, h_idx, bucket_ev, slot_ev, :] = wk_ev
            st.mem_v[b_ev, h_idx, bucket_ev, slot_ev, :] = newv
            st.mem_tag[b_ev, h_idx, bucket_ev, slot_ev, :] = wt_ev
            st.mem_last[b_ev, h_idx, bucket_ev, slot_ev] = time_ev.to(dtype=torch.long)

