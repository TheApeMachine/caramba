"""Memory subsystem for MOSAIC block."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.carmath import last_write_wins
from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic_state import MosaicState


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class MosaicMemory(nn.Module):
    """Hard-addressed associative cache with Bits or VQ routing."""

    def __init__(self, config: MosaicBlockLayerConfig, d_model: int) -> None:
        super().__init__()
        self.config = config

        buckets = int(config.mem_buckets)
        if buckets < 2:
            raise ValueError(f"mem_buckets must be >= 2, got {buckets}")
        hashes = int(config.mem_hashes)
        if hashes < 1:
            raise ValueError(f"mem_hashes must be >= 1, got {hashes}")
        mem_dim = int(config.mem_dim)
        if mem_dim < 1:
            raise ValueError(f"mem_dim must be >= 1, got {mem_dim}")

        self.mem_buckets = buckets
        self.mem_hashes = hashes
        self.mem_dim = mem_dim
        self.mem_router = str(getattr(config, "mem_router", "bits")).lower().strip()
        if self.mem_router not in ("bits", "vq"):
            raise ValueError(f"Unsupported mem_router={self.mem_router!r}")

        # Router setup
        self.mem_bits = int((buckets - 1).bit_length())
        self.mem_read_bits: nn.Linear | None = None
        self.mem_write_bits: nn.Linear | None = None

        # VQ router params
        self.mem_vq_groups = int(getattr(config, "mem_vq_groups", 2))
        self.mem_vq_codebook_size = int(getattr(config, "mem_vq_codebook_size", 256))
        self.mem_vq_group_dim = int(getattr(config, "mem_vq_group_dim", 16))
        self.mem_vq_beam = int(getattr(config, "mem_vq_beam", 1))
        self.mem_write_multi = bool(getattr(config, "mem_write_multi", False))
        self.mem_vq_proj_r: nn.Linear | None = None
        self.mem_vq_proj_w: nn.Linear | None = None
        self.mem_vq_codebook_r: nn.Parameter | None = None
        self.mem_vq_codebook_w: nn.Parameter | None = None

        # Caches for VQ/Bits
        self._vq_powK_cache: Tensor | None = None
        self._vq_combo_idx: Tensor | None = None
        self._bit_pows: Tensor | None = None

        if self.mem_router == "bits":
            self.mem_read_bits = nn.Linear(d_model, hashes * self.mem_bits, bias=False)
            self.mem_write_bits = nn.Linear(d_model, hashes * self.mem_bits, bias=False)
        else:
            G = int(self.mem_vq_groups)
            Kc = int(self.mem_vq_codebook_size)
            gd = int(self.mem_vq_group_dim)
            if G < 1 or Kc < 2 or gd < 1:
                raise ValueError("Invalid VQ router config")
            route_dim = int(G * gd)
            self.mem_vq_proj_r = nn.Linear(d_model, hashes * route_dim, bias=False)
            self.mem_vq_proj_w = nn.Linear(d_model, hashes * route_dim, bias=False)
            self.mem_vq_codebook_r = nn.Parameter(torch.randn(hashes, G, Kc, gd) * 0.02)
            self.mem_vq_codebook_w = nn.Parameter(torch.randn(hashes, G, Kc, gd) * 0.02)
            self.mem_bits = int((buckets - 1).bit_length())

        # Within-bucket fuzzy routing keys (set-associative)
        assoc = int(getattr(config, "mem_assoc", 1))
        if assoc < 1:
            raise ValueError(f"mem_assoc must be >= 1, got {assoc}")
        self.mem_assoc = assoc
        key_dim = int(getattr(config, "mem_key_dim", 0) or 0)
        if key_dim < 1:
            raise ValueError(f"mem_key_dim must be >= 1, got {key_dim}")
        self.mem_key_dim = key_dim

        self.mem_qkey = nn.Linear(d_model, key_dim, bias=False)
        self.mem_wkey = nn.Linear(d_model, key_dim, bias=False)
        self.mem_value = nn.Linear(d_model, mem_dim, bias=False)
        self.mem_out = nn.Linear(mem_dim, d_model, bias=False)
        self.mem_write_gate = nn.Linear(d_model, 1, bias=True)
        self.mem_utility_head = nn.Linear(d_model, 1, bias=True)

    def _vq_powK_tensor(self, device: torch.device) -> Tensor:
        """Return (G,) powers of K for base-K addressing."""
        p = self._vq_powK_cache
        if p is None or p.device != device or p.numel() != int(self.mem_vq_groups):
            Kc = int(self.mem_vq_codebook_size)
            G = int(self.mem_vq_groups)
            p = torch.tensor([Kc**i for i in range(G)], device=device, dtype=torch.long)
            self._vq_powK_cache = p
        return p

    def _vq_combo_index(self, device: torch.device) -> Tensor:
        """Return (Ncand, G) indices selecting beam options per group."""
        beam = int(self.mem_vq_beam)
        G = int(self.mem_vq_groups)
        if beam <= 1:
            idx = torch.zeros((1, G), device=device, dtype=torch.long)
            return idx
        cached = self._vq_combo_idx
        if cached is not None and cached.device == device and cached.shape[1] == G:
            return cached
        grids = torch.cartesian_prod(*([torch.arange(beam, device=device)] * G))
        self._vq_combo_idx = grids.to(dtype=torch.long)
        return self._vq_combo_idx

    def _bit_powers(self, device: torch.device) -> Tensor:
        p = self._bit_pows
        if p is None or p.device != device or p.numel() != self.mem_bits:
            p = (2 ** torch.arange(self.mem_bits, device=device, dtype=torch.long)).view(1, 1, self.mem_bits)
            self._bit_pows = p
        return p

    def _route_bits(self, z: Tensor) -> Tensor:
        """Sign-bit routing: z is (B,H,BITS) -> idx (B,H)."""
        bits = z > 0
        idx = (bits.to(torch.long) * self._bit_powers(z.device)).sum(dim=-1)
        if _is_power_of_two(self.mem_buckets) and (1 << self.mem_bits) == self.mem_buckets:
            return idx
        return torch.remainder(idx, int(self.mem_buckets))

    def _route_vq(
        self,
        y: Tensor,
        *,
        codebook: Tensor,
        return_group_logits: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Product-quantized VQ routing."""
        B, Hh, G, gd = y.shape
        Kc = int(codebook.size(2))
        beam = int(self.mem_vq_beam)

        y2 = (y * y).sum(dim=-1, keepdim=True)
        e = codebook
        e2 = (e * e).sum(dim=-1).view(1, Hh, G, Kc)
        dot = torch.einsum("bhgd,hgkd->bhgk", y, e)
        dist = y2.view(B, Hh, G, 1) + e2 - 2.0 * dot

        group_logits: Tensor | None = None
        if bool(return_group_logits):
            group_logits = (-dist).to(dtype=torch.float32)

        best = dist.argmin(dim=-1)
        powK = self._vq_powK_tensor(y.device).view(1, 1, G)
        idx = (best.to(torch.long) * powK).sum(dim=-1)

        if not (_is_power_of_two(self.mem_buckets) and int(self.mem_buckets) == int(self.mem_vq_codebook_size) ** int(self.mem_vq_groups)):
            idx = torch.remainder(idx, int(self.mem_buckets))

        if beam <= 1:
            return idx, None, None, group_logits

        topk = min(int(beam), Kc)
        top_dist, top_idx = torch.topk(dist, k=topk, dim=-1, largest=False)
        p = torch.softmax((-top_dist).float(), dim=-1).to(dtype=y.dtype)

        combos = self._vq_combo_index(y.device)
        Nc = int(combos.size(0))

        codes_sel = []
        probs_sel = []
        for g in range(G):
            sel = combos[:, g].view(1, 1, Nc).expand(B, Hh, Nc)
            cg = top_idx[:, :, g, :]
            pg = p[:, :, g, :]
            codes_sel.append(torch.gather(cg, dim=-1, index=sel))
            probs_sel.append(torch.gather(pg, dim=-1, index=sel))

        codes = torch.stack(codes_sel, dim=-1)
        probs = torch.stack(probs_sel, dim=-1)
        cand_w = probs.prod(dim=-1)
        cand_idx = (codes.to(torch.long) * powK.view(1, 1, 1, G)).sum(dim=-1)
        cand_idx = torch.remainder(cand_idx, int(self.mem_buckets))
        return idx, cand_idx, cand_w, group_logits

    def compute_routing(self, u: Tensor, collect_aux: bool = False) -> dict[str, Any]:
        """Compute read/write routing indices for input u."""
        B, T, D = u.shape

        results = {}

        if self.mem_router == "bits":
            assert self.mem_read_bits is not None and self.mem_write_bits is not None
            z_read = self.mem_read_bits(u).view(B, T, self.mem_hashes, int(self.mem_bits))
            z_write = self.mem_write_bits(u).view(B, T, self.mem_hashes, int(self.mem_bits))

            idx_r = self._route_bits(z_read.reshape(B * T, self.mem_hashes, int(self.mem_bits))).view(B, T, self.mem_hashes)
            idx_w = self._route_bits(z_write.reshape(B * T, self.mem_hashes, int(self.mem_bits))).view(B, T, self.mem_hashes)

            results["idx_r"] = idx_r
            results["idx_w"] = idx_w
            if collect_aux:
                results["read_bit_logits"] = z_read
                results["write_bit_logits"] = z_write

        else:
            assert self.mem_vq_proj_r is not None and self.mem_vq_proj_w is not None
            assert self.mem_vq_codebook_r is not None and self.mem_vq_codebook_w is not None
            G = int(self.mem_vq_groups)
            gd = int(self.mem_vq_group_dim)
            route_dim = int(G * gd)

            yr = self.mem_vq_proj_r(u).view(B, T, self.mem_hashes, route_dim).view(B * T, self.mem_hashes, G, gd)
            yw = self.mem_vq_proj_w(u).view(B, T, self.mem_hashes, route_dim).view(B * T, self.mem_hashes, G, gd)

            idx_r, cand_r_idx, cand_r_w, gl_r = self._route_vq(
                yr, codebook=self.mem_vq_codebook_r, return_group_logits=bool(collect_aux)
            )
            idx_w, cand_w_idx, cand_w_w, gl_w = self._route_vq(
                yw, codebook=self.mem_vq_codebook_w, return_group_logits=bool(collect_aux)
            )

            results["idx_r"] = idx_r.view(B, T, self.mem_hashes)
            results["idx_w"] = idx_w.view(B, T, self.mem_hashes)

            if isinstance(cand_r_idx, Tensor) and isinstance(cand_r_w, Tensor):
                results["cand_r_idx"] = cand_r_idx.view(B, T, self.mem_hashes, -1)
                results["cand_r_w"] = cand_r_w.view(B, T, self.mem_hashes, -1)
            if isinstance(cand_w_idx, Tensor) and isinstance(cand_w_w, Tensor):
                results["cand_w_idx"] = cand_w_idx.view(B, T, self.mem_hashes, -1)
                results["cand_w_w"] = cand_w_w.view(B, T, self.mem_hashes, -1)

            if collect_aux:
                if isinstance(gl_r, Tensor):
                    results["read_vq_logits"] = gl_r.view(B, T, self.mem_hashes, G, int(self.mem_vq_codebook_size))
                if isinstance(gl_w, Tensor):
                    results["write_vq_logits"] = gl_w.view(B, T, self.mem_hashes, G, int(self.mem_vq_codebook_size))

        return results

    def read(self, u: Tensor, st: MosaicState, routing: dict[str, Any]) -> Tensor:
        """Perform memory read.

        Args:
            u: (B, C, D) input
            st: MosaicState (for mem_k, mem_v, mem_last)
            routing: dict with idx_r, cand_r_idx, cand_r_w

        Returns:
            read_output: (B, C, D)
        """
        B, C, D = u.shape
        Hh = self.mem_hashes
        A = self.mem_assoc
        key_dim = self.mem_key_dim
        mem_dim = self.mem_dim
        temp = float(getattr(self.config, "mem_read_temp", 1.0))
        temp = max(1e-6, temp)
        key_scale = 1.0 / math.sqrt(float(key_dim))

        idx_r = routing["idx_r"]  # (B, C, H)
        cand_r_idx = routing.get("cand_r_idx")
        cand_r_w = routing.get("cand_r_w")

        qk = self.mem_qkey(u)  # (B, C, key_dim)

        mem_k = st.mem_k  # (B, H, buckets, A, key_dim)
        mem_v = st.mem_v
        mem_last = st.mem_last

        # If C>1, we might need to reshape mem tensors if they are (B, H, buckets...) and we want to gather with (B, C, H).
        # Actually gather requires compatible shapes.
        # If we are in chunked mode (C > 1), we use (B, C, H) indices.

        # We need to expand mem tensors to match C dim or reshape indices.
        # Gather strategy:
        # idx_r: (B, C, H) -> (B, H, C) permute
        # Expand mem to (B, H, buckets, A, ...)
        # It's better to use gathers on flat batch or careful broadcasting.

        # Let's align with the original implementation which uses explicit gathers.
        # Original: idx_g = idx_r_c.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) # (B, H, C, 1, 1)
        # mem_k gathered with idx_g expanded.

        if cand_r_idx is None or cand_r_w is None:
            # Single bucket read
            idx_g = idx_r.permute(0, 2, 1).to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)  # (B, H, C, 1, 1)

            # mem_k: (B, H, buckets, A, key_dim)
            # index: (B, H, C, A, key_dim)
            bk = mem_k.gather(dim=2, index=idx_g.expand(B, Hh, C, A, key_dim))
            bv = mem_v.gather(dim=2, index=idx_g.expand(B, Hh, C, A, mem_dim))
            bl = mem_last.gather(dim=2, index=idx_g[..., 0].expand(B, Hh, C, A))

            valid = bl >= 0
            sim = (bk * qk.view(B, 1, C, 1, key_dim)).sum(dim=-1) * float(key_scale)
            sim = sim.masked_fill(~valid, float("-inf"))
            any_valid = valid.any(dim=-1, keepdim=True)
            w = torch.softmax(sim / float(temp), dim=-1)
            w = torch.where(any_valid, w, torch.zeros_like(w))

            read_h = (w.unsqueeze(-1) * bv).sum(dim=3)  # (B,H,C,mem_dim)
            r_c = self.mem_out(read_h.sum(dim=1).permute(0, 1, 2).reshape(B * C, mem_dim)).view(B, C, D)
            # Wait, original summed dim 1 (H). read_h is (B, H, C, mem_dim)
            # read_h.sum(dim=1) -> (B, C, mem_dim)
            r_c = self.mem_out(read_h.sum(dim=1))

        else:
            # Multi-bucket read (VQ beam)
            cand_idx_c = cand_r_idx  # (B, C, H, Nc)
            cand_w_c = cand_r_w      # (B, C, H, Nc)
            ww = cand_w_c / cand_w_c.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            read_h = u.new_zeros((B, Hh, C, mem_dim))
            Nc = int(cand_idx_c.size(-1))

            for ci in range(Nc):
                idx_ci = cand_idx_c[:, :, :, ci]  # (B, C, H)
                idx_g = idx_ci.permute(0, 2, 1).to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)

                bk = mem_k.gather(dim=2, index=idx_g.expand(B, Hh, C, A, key_dim))
                bv = mem_v.gather(dim=2, index=idx_g.expand(B, Hh, C, A, mem_dim))
                bl = mem_last.gather(dim=2, index=idx_g[..., 0].expand(B, Hh, C, A))

                valid = bl >= 0
                sim = (bk * qk.view(B, 1, C, 1, key_dim)).sum(dim=-1) * float(key_scale)
                sim = sim.masked_fill(~valid, float("-inf"))
                any_valid = valid.any(dim=-1, keepdim=True)
                wslot = torch.softmax(sim / float(temp), dim=-1)
                wslot = torch.where(any_valid, wslot, torch.zeros_like(wslot))
                rh_ci = (wslot.unsqueeze(-1) * bv).sum(dim=3)  # (B, H, C, mem_dim)

                # ww is (B, C, H, Nc), take slice ci -> (B, C, H) -> permute to (B, H, C)
                w_ci = ww[:, :, :, ci].permute(0, 2, 1).unsqueeze(-1)
                read_h = read_h + rh_ci * w_ci

            r_c = self.mem_out(read_h.sum(dim=1))

        return r_c

    def write_chunk(self, u: Tensor, st: MosaicState, routing: dict[str, Any], t0: int, mask: Tensor | None = None) -> Tensor:
        """Perform memory write for a chunk of data.

        Args:
            u: (B, C, D) input chunk
            st: MosaicState
            routing: dict with idx_w, etc. (already sliced for this chunk)
            t0: start time offset for this chunk
            mask: optional (B, C) mask to force/inhibit writes

        Returns:
            gate_logits: (B, C) for aux loss
        """
        B, C, D = u.shape
        Hh = self.mem_hashes
        A = self.mem_assoc
        key_dim = self.mem_key_dim
        mem_dim = self.mem_dim
        eta = float(self.config.mem_write_eta)
        thr = float(self.config.mem_write_threshold)
        match_thr = float(getattr(self.config, "mem_match_threshold", 0.0))
        key_scale = 1.0 / math.sqrt(float(key_dim))

        idx_w = routing["idx_w"]  # (B, C, H)
        cand_w_idx = routing.get("cand_w_idx")
        cand_w_w = routing.get("cand_w_w")

        gate_logit = self.mem_write_gate(u).squeeze(-1)  # (B, C)
        p = torch.sigmoid(gate_logit)

        m = (p > thr).to(dtype=u.dtype)
        if mask is not None:
             m = torch.where(mask >= 0, (mask > 0).to(dtype=m.dtype, device=m.device), m)

        w_eta = (float(eta) * p).to(dtype=u.dtype) * m

        do = w_eta > 0
        if not do.any():
            return gate_logit

        pos = torch.nonzero(do, as_tuple=False)
        if pos.numel() == 0:
             return gate_logit

        b_ev_all = pos[:, 0]
        t_ev_all = pos[:, 1]

        wk = self.mem_wkey(u)  # (B, C, key_dim)
        v = self.mem_value(u)  # (B, C, mem_dim)

        # Collect write bucket indices per hash head
        for h in range(Hh):
            write_buckets_h: list[Tensor] = [idx_w[:, :, h].to(dtype=torch.long)]

            # If multi-write is enabled with VQ
            if (self.mem_router == "vq"
                and self.mem_write_multi
                and cand_w_idx is not None
                and cand_w_w is not None):
                cw = cand_w_w[:, :, h, :]
                ci = cand_w_idx[:, :, h, :]
                k2 = min(2, int(cw.size(-1)))
                if k2 >= 2:
                    top2 = torch.topk(cw, k=int(k2), dim=-1).indices
                    b1 = torch.gather(ci, dim=-1, index=top2[:, :, 1:2]).squeeze(-1)
                    write_buckets_h.append(b1.to(dtype=torch.long))

            for bidx in write_buckets_h:
                # bidx: (B, C) bucket index for head h

                # Fetch current memory content
                mk = st.mem_k[:, h, :, :, :]  # (B, buckets, A, key_dim)
                mv = st.mem_v[:, h, :, :, :]
                ml = st.mem_last[:, h, :, :]

                idxk = bidx.unsqueeze(-1).unsqueeze(-1).expand(B, C, A, key_dim)
                idxv = bidx.unsqueeze(-1).unsqueeze(-1).expand(B, C, A, mem_dim)
                idxl = bidx.unsqueeze(-1).expand(B, C, A)

                bk_w = mk.gather(dim=1, index=idxk)
                bv_w = mv.gather(dim=1, index=idxv)
                bl_w = ml.gather(dim=1, index=idxl)

                valid_w = bl_w >= 0
                sim_w = (bk_w * wk.unsqueeze(2)).sum(dim=-1) * float(key_scale)
                sim_w = sim_w.masked_fill(~valid_w, float("-inf"))
                best_slot = sim_w.argmax(dim=-1)
                best_sim = sim_w.max(dim=-1).values
                has_empty = (~valid_w).any(dim=-1)
                first_empty = (~valid_w).to(torch.int64).argmax(dim=-1)
                lru_slot = bl_w.argmin(dim=-1)
                repl_slot = torch.where(has_empty, first_empty, lru_slot)

                use_update = torch.isfinite(best_sim) & (best_sim >= float(match_thr)) & has_empty.logical_not()
                slot = torch.where(use_update, best_slot, repl_slot).to(dtype=torch.long)

                # Extract event info
                bucket_ev = bidx[b_ev_all, t_ev_all]
                slot_ev = slot[b_ev_all, t_ev_all]
                eta_ev = w_eta[b_ev_all, t_ev_all].to(dtype=u.dtype)
                wk_ev = wk[b_ev_all, t_ev_all, :].to(dtype=u.dtype)
                v_ev = v[b_ev_all, t_ev_all, :].to(dtype=u.dtype)
                upd_ev = use_update[b_ev_all, t_ev_all]
                time_ev = (int(st.step) + t0) + t_ev_all.to(torch.long)

                # Global key for last_write_wins
                # Key = (batch * H + h) * buckets * assoc + bucket * assoc + slot
                key_ev = (((b_ev_all.to(torch.long) * Hh + int(h)) * int(self.mem_buckets) + bucket_ev) * A + slot_ev)

                winner = last_write_wins(key_ev, time_ev, big=int(st.step + 100000)) # big just needs to be larger than any time

                bw = b_ev_all[winner]
                buckw = bucket_ev[winner]
                slotw = slot_ev[winner]
                etaw = eta_ev[winner].view(-1, 1)
                wkw = wk_ev[winner]
                vw = v_ev[winner]
                updw = upd_ev[winner].view(-1, 1)
                timew = time_ev[winner]

                curk = st.mem_k[bw, h, buckw, slotw, :]
                curv = st.mem_v[bw, h, buckw, slotw, :]

                with torch.no_grad():
                    st.mem_k[bw, h, buckw, slotw, :] = torch.where(
                        updw, (1.0 - etaw) * curk + etaw * wkw, wkw
                    )
                    st.mem_v[bw, h, buckw, slotw, :] = torch.where(
                        updw, (1.0 - etaw) * curv + etaw * vw, vw
                    )
                    st.mem_last[bw, h, buckw, slotw] = timew

        return gate_logit

