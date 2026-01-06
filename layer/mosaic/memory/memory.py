"""MOSAIC memory module

This module implements a hard-addressed, set-associative memory: you route a
query into buckets, perform a match within each bucket, and then read/write
values with explicit update rules instead of implicit “memory in weights”.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.memory.reader import MemoryReader
from caramba.layer.mosaic.memory.routing import BitRouter, VqRouter
from caramba.layer.mosaic.memory.phase import PhaseSimilarity, PhaseTagProjector
from caramba.layer.mosaic.memory.rmf import ResonantMemoryField
from caramba.layer.mosaic.memory.vsa import VsaNovelty, VsaTagProjector
from caramba.layer.mosaic.memory.writer import MemoryWriter
from caramba.layer.mosaic.state import MosaicState


class MosaicMemory(nn.Module):
    """MOSAIC memory subsystem

    Explicit memory makes retrieval and update a first-class, inspectable part
    of the model, which is useful when you want to study credit assignment,
    storage policies, or long-horizon behavior.
    """

    def __init__(self, config: MosaicBlockLayerConfig, d_model: int) -> None:
        super().__init__()
        self.config = config
        self.mem_index = str(getattr(config, "mem_index", "hash")).lower().strip()
        self.mem_buckets = int(config.mem_buckets)  # leaf buckets (routing space)
        self.mem_table_buckets = int(self.mem_buckets)  # physical table buckets (state space)
        self.mem_hashes = int(config.mem_hashes)
        self.mem_dim = int(config.mem_dim)
        self.mem_assoc = int(getattr(config, "mem_assoc", 1))
        self.mem_key_dim = int(getattr(config, "mem_key_dim", 32))
        self.mem_router = str(getattr(config, "mem_router", "bits")).lower().strip()
        self.mem_read_temp = float(getattr(config, "mem_read_temp", 1.0))
        self.mem_write_threshold = float(getattr(config, "mem_write_threshold", 0.5))
        self.mem_write_eta = float(getattr(config, "mem_write_eta", 0.1))
        self.mem_match_threshold = float(getattr(config, "mem_match_threshold", 0.0))
        self.mem_vsa_enabled = bool(getattr(config, "mem_vsa_enabled", True))
        self.mem_vsa_dim = int(getattr(config, "mem_vsa_dim", 32))
        self.mem_vsa_weight = float(getattr(config, "mem_vsa_weight", 1.0))
        self.mem_vsa_tanh_scale = float(getattr(config, "mem_vsa_tanh_scale", 1.0))
        self.mem_vsa_novelty_beta = float(getattr(config, "mem_vsa_novelty_beta", 1.0))
        self.mem_vsa_novelty_threshold = float(getattr(config, "mem_vsa_novelty_threshold", 0.0))
        self.mem_phase_enabled = bool(getattr(config, "mem_phase_enabled", False))
        self.mem_phase_dim = int(getattr(config, "mem_phase_dim", self.mem_key_dim))
        self.mem_phase_weight = float(getattr(config, "mem_phase_weight", 0.0))
        self.mem_phase_tanh_scale = float(getattr(config, "mem_phase_tanh_scale", 1.0))
        self.mem_trie_eta_decay = float(getattr(config, "mem_trie_eta_decay", 0.5))
        self.mem_trie_max_levels = getattr(config, "mem_trie_max_levels", None)
        self.mem_trie_fallback_enabled = bool(getattr(config, "mem_trie_fallback_enabled", True))
        self.rmf_enabled = bool(getattr(config, "rmf_enabled", False))
        self.rmf_dim = int(getattr(config, "rmf_dim", 64))
        self.rmf_eta = float(getattr(config, "rmf_eta", 0.2))
        self.rmf_weight = float(getattr(config, "rmf_weight", 1.0))

        if self.mem_index not in {"hash", "trie"}:
            raise ValueError(f"mem_index must be 'hash' or 'trie', got {self.mem_index!r}")

        if self.mem_buckets < 2:
            raise ValueError("mem_buckets must be >= 2")
        if self.mem_index == "trie":
            # Flat binary radix trie layout:
            # - leaf buckets are addressed by the router (0..L-1)
            # - physical table stores internal prefix nodes too (2L-1 buckets).
            if self.mem_buckets & (self.mem_buckets - 1) != 0:
                raise ValueError("mem_index='trie' requires mem_buckets to be a power of two (leaf count)")
            self.mem_table_buckets = int(2 * self.mem_buckets - 1)
        if self.mem_hashes < 1:
            raise ValueError("mem_hashes must be >= 1")
        if self.mem_dim < 1:
            raise ValueError("mem_dim must be >= 1")
        if self.mem_assoc < 1:
            raise ValueError("mem_assoc must be >= 1")
        if self.mem_key_dim < 1:
            raise ValueError("mem_key_dim must be >= 1")
        if self.mem_vsa_dim < 1:
            raise ValueError("mem_vsa_dim must be >= 1")
        if self.mem_vsa_weight < 0.0:
            raise ValueError("mem_vsa_weight must be >= 0")
        if self.mem_vsa_tanh_scale <= 0.0:
            raise ValueError("mem_vsa_tanh_scale must be > 0")
        if self.mem_vsa_novelty_beta <= 0.0:
            raise ValueError("mem_vsa_novelty_beta must be > 0")
        if self.mem_phase_dim < 1:
            raise ValueError("mem_phase_dim must be >= 1")
        if self.mem_phase_weight < 0.0:
            raise ValueError("mem_phase_weight must be >= 0")
        if self.mem_phase_tanh_scale <= 0.0:
            raise ValueError("mem_phase_tanh_scale must be > 0")
        if self.mem_trie_eta_decay < 0.0 or self.mem_trie_eta_decay > 1.0:
            raise ValueError("mem_trie_eta_decay must be in [0,1]")
        if self.rmf_dim < 1:
            raise ValueError("rmf_dim must be >= 1")
        if self.rmf_eta < 0.0 or self.rmf_eta > 1.0:
            raise ValueError("rmf_eta must be in [0,1]")
        if self.rmf_weight < 0.0:
            raise ValueError("rmf_weight must be >= 0")

        self.mem_qkey = nn.Linear(int(d_model), int(self.mem_key_dim), bias=False)
        self.mem_wkey = nn.Linear(int(d_model), int(self.mem_key_dim), bias=False)
        self.mem_value = nn.Linear(int(d_model), int(self.mem_dim), bias=False)
        self.mem_out = nn.Linear(int(self.mem_dim), int(d_model), bias=False)
        self.mem_write_gate = nn.Linear(int(d_model), 1, bias=True)
        self.mem_utility_head = nn.Linear(int(d_model), 1, bias=True)

        self.vsa_projector: VsaTagProjector | None = None
        self.vsa_novelty: VsaNovelty | None = None
        if self.mem_vsa_enabled:
            self.vsa_projector = VsaTagProjector(
                in_dim=int(self.mem_key_dim),
                vsa_dim=int(self.mem_vsa_dim),
                tanh_scale=float(self.mem_vsa_tanh_scale),
            )
            self.vsa_novelty = VsaNovelty(
                beta=float(self.mem_vsa_novelty_beta),
                threshold=float(self.mem_vsa_novelty_threshold),
            )

        self.phase_projector: PhaseTagProjector | None = None
        self.phase_similarity: PhaseSimilarity | None = None
        if self.mem_phase_enabled and float(self.mem_phase_weight) != 0.0:
            self.phase_projector = PhaseTagProjector(
                in_dim=int(self.mem_key_dim),
                phase_dim=int(self.mem_phase_dim),
                tanh_scale=float(self.mem_phase_tanh_scale),
            )
            self.phase_similarity = PhaseSimilarity(phase_dim=int(self.mem_phase_dim))

        self.rmf: ResonantMemoryField | None = None
        if self.rmf_enabled and float(self.rmf_weight) != 0.0:
            self.rmf = ResonantMemoryField(
                buckets=int(self.mem_buckets),
                rmf_dim=int(self.rmf_dim),
                key_dim=int(self.mem_key_dim),
                eta=float(self.rmf_eta),
            )

        self.bit_router: BitRouter | None = None
        self.vq_router: VqRouter | None = None
        if self.mem_router == "bits":
            self.bit_router = BitRouter(in_dim=int(self.mem_key_dim), hashes=int(self.mem_hashes), buckets=int(self.mem_buckets))
        else:
            self.vq_router = VqRouter(
                in_dim=int(self.mem_key_dim),
                hashes=int(self.mem_hashes),
                buckets=int(self.mem_buckets),
                groups=int(getattr(self.config, "mem_vq_groups", 2)),
                codebook_size=int(getattr(self.config, "mem_vq_codebook_size", 256)),
                group_dim=int(getattr(self.config, "mem_vq_group_dim", 16)),
                beam=int(getattr(self.config, "mem_vq_beam", 1)),
                write_multi=bool(getattr(self.config, "mem_write_multi", False)),
            )
        self.reader = MemoryReader(
            mem_out=self.mem_out,
            mem_qkey=self.mem_qkey,
            mem_key_dim=int(self.mem_key_dim),
            mem_dim=int(self.mem_dim),
            mem_tag_dim=int(self.mem_vsa_dim),
            mem_assoc=int(self.mem_assoc),
            mem_hashes=int(self.mem_hashes),
            mem_buckets=int(self.mem_buckets),
            mem_table_buckets=int(self.mem_table_buckets),
            mem_read_temp=float(self.mem_read_temp),
            mem_vsa_weight=float(self.mem_vsa_weight),
            mem_vsa_enabled=bool(self.mem_vsa_enabled),
            vsa_projector=self.vsa_projector,
            mem_phase_weight=float(self.mem_phase_weight),
            mem_phase_enabled=bool(self.mem_phase_enabled),
            phase_projector=self.phase_projector,
            phase_similarity=self.phase_similarity,
            mem_trie_enabled=bool(self.mem_index == "trie"),
            mem_trie_fallback_enabled=bool(self.mem_trie_fallback_enabled),
            mem_trie_max_levels=self.mem_trie_max_levels,
        )
        self.writer = MemoryWriter(
            mem_wkey=self.mem_wkey,
            mem_value=self.mem_value,
            mem_write_gate=self.mem_write_gate,
            mem_buckets=int(self.mem_buckets),
            mem_table_buckets=int(self.mem_table_buckets),
            mem_hashes=int(self.mem_hashes),
            mem_assoc=int(self.mem_assoc),
            mem_key_dim=int(self.mem_key_dim),
            mem_tag_dim=int(self.mem_vsa_dim),
            mem_dim=int(self.mem_dim),
            mem_write_threshold=float(self.mem_write_threshold),
            mem_write_eta=float(self.mem_write_eta),
            mem_match_threshold=float(self.mem_match_threshold),
            mem_vsa_enabled=bool(self.mem_vsa_enabled),
            vsa_projector=self.vsa_projector,
            vsa_novelty=self.vsa_novelty,
            mem_trie_enabled=bool(self.mem_index == "trie"),
            mem_trie_eta_decay=float(self.mem_trie_eta_decay),
            mem_trie_max_levels=int(self.mem_trie_max_levels) if self.mem_trie_max_levels is not None else None,
        )

    def compute_routing(self, u: Tensor, *, collect_aux: bool) -> dict[str, Any]:
        """Compute routing indices for a sequence

        Routing reduces continuous tags to discrete bucket indices, which is the
        key step that makes reads/writes constant-time with respect to memory
        capacity.
        """
        tag = self.mem_qkey(u).to(dtype=u.dtype)
        if self.mem_router == "bits":
            if self.bit_router is None:
                raise RuntimeError("bit_router is None but mem_router='bits'")
            routing = self.bit_router.route(tag=tag, collect_aux=collect_aux)
        else:
            if self.vq_router is None:
                raise RuntimeError("vq_router is None but mem_router='vq'")
            routing = self.vq_router.route(tag=tag, collect_aux=collect_aux)
        out: dict[str, Any] = {"idx_r": routing.idx_r, "idx_w": routing.idx_w}
        out.update(routing.aux)
        return out

    def compute_routing_with_teacher(self, u: Tensor, st: MosaicState, teacher: dict[str, Tensor], *, collect_aux: bool) -> dict[str, Any]:
        """Compute routing with teacher-assisted RMF

        When teacher read buckets are available, RMF can track a successor-biased
        field and inject a routing delta, which makes it easier to study how
        control signals shape memory access patterns.
        """
        tag = self.mem_qkey(u).to(dtype=u.dtype)
        aux: dict[str, Tensor] = {}
        if self.rmf is not None and "read_bucket" in teacher:
            rb = teacher["read_bucket"]
            hist = self.rmf.field_history(idx_r=rb, device=u.device, dtype=u.dtype)
            st.rmf_field = hist[:, -1].detach()
            flat = hist.reshape(int(u.size(0)) * int(u.size(1)), 2 * int(self.rmf_dim))
            delta = self.rmf.bias(flat).view(int(u.size(0)), int(u.size(1)), int(self.mem_key_dim))
            tag = tag + float(self.rmf_weight) * delta.to(dtype=u.dtype)
            if collect_aux:
                aux["rmf_delta_rms"] = torch.sqrt((delta * delta).mean(dim=-1).clamp_min(0.0)).detach()
                aux["rmf_field_rms"] = torch.sqrt((hist[:, -1, :, 0] ** 2 + hist[:, -1, :, 1] ** 2).mean(dim=1).clamp_min(0.0)).view(int(u.size(0)), 1).detach()
        out = self.route_from_tag(tag=tag, collect_aux=collect_aux)
        out.update(aux)
        return out

    def route_from_tag(self, *, tag: Tensor, collect_aux: bool) -> dict[str, Any]:
        """Route using a precomputed tag

        Separating “compute tag” from “route tag” lets other modules (like RMF)
        perturb tags before routing without needing to reimplement router logic.
        """
        if self.mem_router == "bits":
            if self.bit_router is None:
                raise RuntimeError("bit_router is None but mem_router='bits'")
            r = self.bit_router.route(tag=tag, collect_aux=collect_aux)
        else:
            if self.vq_router is None:
                raise RuntimeError("vq_router is None but mem_router='vq'")
            r = self.vq_router.route(tag=tag, collect_aux=collect_aux)
        out: dict[str, Any] = {"idx_r": r.idx_r, "idx_w": r.idx_w}
        out.update(r.aux)
        return out

    def apply_teacher_overrides(self, routing: dict[str, Any], teacher: dict[str, Tensor], *, p: float) -> None:
        """Apply teacher bucket overrides

        Teacher forcing for routing is a research tool: it can stabilize early
        training and lets you measure how well a learned router matches a known
        addressing policy.
        """
        if not (0.0 < float(p) <= 1.0):
            return
        idx_r = routing.get("idx_r", None)
        idx_w = routing.get("idx_w", None)
        if not (isinstance(idx_r, Tensor) and isinstance(idx_w, Tensor)):
            raise TypeError("routing must include idx_r and idx_w tensors")
        B, T, H = int(idx_r.size(0)), int(idx_r.size(1)), int(idx_r.size(2))
        m = (torch.rand((B, T, 1), device=idx_r.device) < float(p)).expand(B, T, H)
        self.override_with_stats(routing, teacher, m=m, key="read_bucket", name="idx_r", stats_key="read_teacher_agree")
        self.override_with_stats(routing, teacher, m=m, key="write_bucket", name="idx_w", stats_key="write_teacher_agree")
        routing["teacher_used_frac"] = m.detach().float().mean()
        if bool(routing.get("collect_aux", False)):
            self.add_vq_accuracy(routing, teacher)

    def override_with_stats(
        self,
        routing: dict[str, Any],
        teacher: dict[str, Tensor],
        *,
        m: Tensor,
        key: str,
        name: str,
        stats_key: str,
    ) -> None:
        if key not in teacher:
            return
        t = teacher[key]
        idx = routing[name]
        pre = idx
        routing[f"{name}_pre"] = pre.detach()
        if t.ndim == 2:
            t = t.unsqueeze(-1).expand_as(idx)
        elif t.ndim == 3:
            if int(t.size(-1)) == 1 and int(idx.size(-1)) > 1:
                t = t.expand_as(idx)
        else:
            raise ValueError(f"teacher[{key!r}] must be (B,T) or (B,T,H), got {tuple(t.shape)}")
        valid = t >= 0
        use = m & valid
        routing[name] = torch.where(use, t.to(dtype=torch.long), idx.to(dtype=torch.long))
        agree_all, agree_free, label_count, probe_count = self.teacher_agreement(pre=pre, teacher=t, use=use)
        routing[stats_key] = agree_all
        routing[f"{stats_key}_free"] = agree_free
        routing[f"{stats_key}_label_count"] = label_count
        routing[f"{stats_key}_probe_count"] = probe_count

    def teacher_agreement(self, *, pre: Tensor, teacher: Tensor, use: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Agreement stats for teacher addressing.

        Returns:
            (agree_all, agree_free, label_count, probe_count) where:
            - agree_all: agreement on all teacher-labeled positions
            - agree_free: agreement on teacher-labeled positions where teacher was NOT applied
            - label_count: number of teacher-labeled positions (hash-0)
            - probe_count: number of probe positions (teacher-labeled AND not forced)
        """
        if pre.ndim != 3 or teacher.ndim != 3 or use.ndim != 3:
            raise ValueError("pre/teacher/use must be (B,T,H)")
        pre0 = pre[:, :, 0]
        t0 = teacher[:, :, 0]
        u0 = use[:, :, 0]
        label = t0 >= 0
        probe = label & (~u0)
        label_count = label.detach().float().sum()
        probe_count = probe.detach().float().sum()
        nan = torch.full_like(label_count, float("nan"))

        agree_all = ((pre0 == t0) & label).detach().float().sum() / label_count.clamp_min(1.0)
        agree_all = torch.where(label_count > 0, agree_all, nan)
        agree_free = ((pre0 == t0) & probe).detach().float().sum() / probe_count.clamp_min(1.0)
        agree_free = torch.where(probe_count > 0, agree_free, nan)
        return (
            agree_all.to(dtype=torch.float32),
            agree_free.to(dtype=torch.float32),
            label_count.to(dtype=torch.float32),
            probe_count.to(dtype=torch.float32),
        )

    def add_vq_accuracy(self, routing: dict[str, Any], teacher: dict[str, Tensor]) -> None:
        if self.vq_router is None:
            return
        K = int(getattr(self.vq_router, "codebook_size", 0))
        G = int(getattr(self.vq_router, "groups", 0))
        if K < 2 or G < 1:
            return

        vqr = routing.get("read_vq_logits", None)
        tr = teacher.get("read_bucket", None)
        if isinstance(vqr, Tensor) and isinstance(tr, Tensor):
            ga, ba = self.vq_accuracy(t_bucket=tr, vq_logits=vqr, codebook_size=K, groups=G)
            routing["vq_read_group_acc"] = ga
            routing["vq_read_bucket_acc"] = ba

        vqw = routing.get("write_vq_logits", None)
        tw = teacher.get("write_bucket", None)
        if isinstance(vqw, Tensor) and isinstance(tw, Tensor):
            ga2, ba2 = self.vq_accuracy(t_bucket=tw, vq_logits=vqw, codebook_size=K, groups=G)
            routing["vq_write_group_acc"] = ga2
            routing["vq_write_bucket_acc"] = ba2

    def vq_accuracy(self, *, t_bucket: Tensor, vq_logits: Tensor, codebook_size: int, groups: int) -> tuple[Tensor, Tensor]:
        if vq_logits.ndim != 5:
            raise ValueError(f"vq_logits must have shape (B,T,H,G,K), got {tuple(vq_logits.shape)}")
        B, T, H, G, K = vq_logits.shape
        if int(K) != int(codebook_size) or int(G) != int(groups):
            raise ValueError("vq_logits shape does not match (codebook_size, groups)")
        tb = t_bucket
        if tb.ndim == 2:
            tb = tb.unsqueeze(-1).expand(B, T, H)
        if tb.ndim != 3 or tb.shape[:3] != (B, T, H):
            raise ValueError(f"t_bucket must have shape (B,T) or (B,T,H), got {tuple(t_bucket.shape)}")
        mask = tb >= 0
        denom = mask.detach().float().sum().clamp_min(1.0)
        pred = vq_logits.detach().argmax(dim=-1)  # (B,T,H,G)
        tgt = tb.detach().long().clamp_min(0)
        codes: list[Tensor] = []
        for g in range(int(G)):
            codes.append(((tgt // (int(K) ** int(g))) % int(K)).unsqueeze(-1))
        tgt_codes = torch.cat(codes, dim=-1)  # (B,T,H,G)
        correct = (pred == tgt_codes) & mask.unsqueeze(-1)
        group_acc = correct.detach().float().sum(dim=(0, 1, 2)) / denom
        group_acc_mean = group_acc.mean()
        bucket_acc = (correct.all(dim=-1) & mask).detach().float().sum() / denom
        return group_acc_mean.to(dtype=torch.float32), bucket_acc.to(dtype=torch.float32)

    def compute_routing_step(self, u: Tensor, st: MosaicState, *, collect_aux: bool) -> dict[str, Any]:
        """Compute routing for a single step/chunk (B,1,D), with optional RMF bias."""
        if u.ndim != 3 or int(u.size(1)) < 1:
            raise ValueError(f"u must have shape (B,T,D) with T>=1, got {tuple(u.shape)}")
        tag = self.mem_qkey(u).to(dtype=u.dtype)
        rmf_aux: dict[str, Tensor] = {}
        if self.rmf is not None:
            st.rmf_field = self.ensure_rmf_field(st, batch=int(u.size(0)), device=u.device, dtype=u.dtype)
            delta = self.rmf.routing_delta(field=st.rmf_field).view(int(u.size(0)), 1, int(self.mem_key_dim))
            tag = tag + float(self.rmf_weight) * delta.to(dtype=u.dtype)
            if collect_aux:
                d_rms = torch.sqrt((delta * delta).mean(dim=-1).clamp_min(0.0))
                rmf_aux["rmf_delta_rms"] = d_rms.detach()
        if self.mem_router == "bits":
            if self.bit_router is None:
                raise RuntimeError("bit_router is None but mem_router='bits'")
            routing = self.bit_router.route(tag=tag, collect_aux=collect_aux)
        else:
            if self.vq_router is None:
                raise RuntimeError("vq_router is None but mem_router='vq'")
            routing = self.vq_router.route(tag=tag, collect_aux=collect_aux)
        out: dict[str, Any] = {"idx_r": routing.idx_r, "idx_w": routing.idx_w}
        out.update(rmf_aux)
        out.update(routing.aux)
        return out

    def update_rmf(self, st: MosaicState, routing: dict[str, Any]) -> None:
        """Update RMF state from routing outputs."""
        if self.rmf is None:
            return
        idx_r = routing.get("idx_r", None)
        if not isinstance(idx_r, Tensor):
            raise TypeError("routing must contain Tensor idx_r when RMF is enabled")
        st.rmf_field = self.ensure_rmf_field(st, batch=int(idx_r.size(0)), device=idx_r.device, dtype=st.mem_k.dtype)
        st.rmf_field = self.rmf.update(field=st.rmf_field, idx_r=idx_r)
        if bool(routing.get("collect_aux", False)):
            f = st.rmf_field
            re = f[:, :, 0]
            im = f[:, :, 1]
            f_rms = torch.sqrt((re * re + im * im).mean(dim=1).clamp_min(0.0)).view(int(f.size(0)), 1)
            routing["rmf_field_rms"] = f_rms.detach()

    def ensure_rmf_field(self, st: MosaicState, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.rmf is None:
            raise RuntimeError("RMF requested but rmf is None")
        f = st.rmf_field
        if isinstance(f, Tensor) and tuple(f.shape) == (int(batch), int(self.rmf_dim), 2) and f.device == device:
            return f.to(dtype=dtype)
        return self.rmf.initial_field(batch=int(batch), device=device, dtype=dtype)

    def read(self, u: Tensor, st: MosaicState, routing: dict[str, Any]) -> Tensor:
        return self.reader.read(u, st, routing)

    def write_chunk(
        self,
        u: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        t0: int,
        mask: Tensor | None,
        *,
        write_scale: Tensor | None = None,
    ) -> Tensor:
        return self.writer.write_chunk(u, st, routing, int(t0), mask, write_scale)

