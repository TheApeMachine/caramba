from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class BridgeOutput:
    freq_logits: torch.Tensor
    freq_probs: torch.Tensor
    meta: Dict[str, Any]


class BipartiteBondGraph:
    """Sparse bonds from semantic sources -> spectral destinations."""

    def __init__(self, num_src: int, num_dst: int, *, device: torch.device, dtype: torch.dtype, eps: float):
        self.num_src = int(num_src)
        self.num_dst = int(num_dst)
        self.device = device
        self.dtype = dtype
        self.eps = float(eps)

        self.src = torch.empty(0, dtype=torch.long, device=device)
        self.dst = torch.empty(0, dtype=torch.long, device=device)
        self.w = torch.empty(0, dtype=dtype, device=device)
        self.trace = torch.empty(0, dtype=dtype, device=device)

        self._edge_index: Dict[Tuple[int, int], int] = {}
        self._out_edges: Dict[int, list[int]] = {}

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def add_edge(self, s: int, d: int, mass: torch.Tensor) -> None:
        s = int(s)
        d = int(d)
        key = (s, d)
        if key in self._edge_index:
            ei = self._edge_index[key]
            self.w[ei] = self.w[ei] + mass.to(self.dtype)
            return

        ei = self.num_edges
        self._edge_index[key] = ei
        self._out_edges.setdefault(s, []).append(ei)

        self.src = torch.cat([self.src, torch.tensor([s], device=self.device, dtype=torch.long)], dim=0)
        self.dst = torch.cat([self.dst, torch.tensor([d], device=self.device, dtype=torch.long)], dim=0)
        self.w = torch.cat([self.w, mass.to(self.dtype).view(1)], dim=0)
        self.trace = torch.cat([self.trace, torch.zeros(1, device=self.device, dtype=self.dtype)], dim=0)

    def edges_from(self, s: int) -> list[int]:
        return self._out_edges.get(int(s), [])

    def flow(self, src_dist: torch.Tensor) -> torch.Tensor:
        """Push a source distribution through the bipartite graph."""
        if self.num_edges == 0:
            return torch.zeros(self.num_dst, device=self.device, dtype=self.dtype)

        out_sum = torch.zeros(self.num_src, device=self.device, dtype=self.dtype)
        out_sum.index_add_(0, self.src, self.w)
        w_norm = self.w / (out_sum[self.src] + self.eps)

        contrib = src_dist[self.src] * w_norm
        dst_mass = torch.zeros(self.num_dst, device=self.device, dtype=self.dtype)
        dst_mass.index_add_(0, self.dst, contrib)
        return dst_mass

    def metabolize(self, active_src: torch.Tensor, src_act: torch.Tensor, dst_act: torch.Tensor, *, dt: float, ratio: torch.Tensor) -> None:
        """Local bond dynamics on active sources."""
        if active_src.numel() == 0 or self.num_edges == 0:
            return
        eps = self.eps

        # Build active edge index list.
        idx = []
        for s in active_src.tolist():
            idx.extend(self.edges_from(int(s)))
        if not idx:
            return
        eidx = torch.tensor(idx, device=self.device, dtype=torch.long)

        s = self.src[eidx]
        d = self.dst[eidx]
        w = self.w[eidx]
        tr = self.trace[eidx]

        # Normalization per source (local).
        out_sum = torch.zeros(self.num_src, device=self.device, dtype=self.dtype)
        out_sum.index_add_(0, s, w)
        w_norm = w / (out_sum[s] + eps)

        use = src_act[s] * dst_act[d] * w_norm

        tr_scale = tr.abs().mean() + eps
        use_scale = use.abs().mean() + eps
        decay = torch.exp(-dt * ratio / (tr_scale + use_scale))
        tr_new = tr * decay + use

        cost = ratio * use_scale * w / (w.abs().mean() + eps)
        w_new = (w + dt * (use - cost)).clamp(min=0.0)

        self.w[eidx] = w_new
        self.trace[eidx] = tr_new


class BridgeManifold:
    """Emergent transducer bonds between semantic activation and spectral activation."""

    def __init__(self, *, num_semantic: int, num_spectral: int, dt: float, device: torch.device, eps: float = 1e-8):
        self.device = device
        self.dt = float(dt)
        self.eps = float(eps)

        self.num_semantic = int(num_semantic)
        self.num_spectral = int(num_spectral)

        self.graph = BipartiteBondGraph(num_semantic, num_spectral, device=device, dtype=torch.float32, eps=eps)

        self._energy_baseline: Optional[torch.Tensor] = None

    def _ratio(self, total: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        eps = self.eps
        if self._energy_baseline is None:
            self._energy_baseline = total.detach().clone()
            return torch.tensor(1.0, device=self.device)
        base = self._energy_baseline
        alpha = dt / (dt + float(base.abs().item()) + eps)
        base_new = base * (1.0 - alpha) + total.detach() * alpha
        self._energy_baseline = base_new
        return total / (base_new + eps)

    def coactivate(self, sem_ids: torch.Tensor, sem_act: torch.Tensor, spec_ids: torch.Tensor, spec_act: torch.Tensor) -> None:
        """Nucleate bonds based on observed co-activation (data-driven, not hard-coded)."""
        sem_ids = sem_ids.to(device=self.device, dtype=torch.long).flatten()
        spec_ids = spec_ids.to(device=self.device, dtype=torch.long).flatten()
        if sem_ids.numel() == 0 or spec_ids.numel() == 0:
            return
        dt = self.dt
        eps = self.eps

        s_scale = sem_act[sem_ids].abs().mean() + eps
        d_scale = spec_act[spec_ids].abs().mean() + eps
        # Outer-product nucleation over active sets (typically small).
        for s in sem_ids.tolist():
            for d in spec_ids.tolist():
                mass = dt * (sem_act[s] / s_scale) * (spec_act[d] / d_scale)
                self.graph.add_edge(int(s), int(d), mass.to(torch.float32))

        # Metabolize the just-activated sources.
        total = sem_act.abs().sum() + spec_act.abs().sum()
        ratio = self._ratio(total.to(torch.float32))
        self.graph.metabolize(sem_ids.unique(), sem_act, spec_act, dt=dt, ratio=ratio.to(torch.float32))

    def forward(self, sem_dist: torch.Tensor) -> BridgeOutput:
        sem_dist = sem_dist.to(device=self.device, dtype=torch.float32).flatten()
        sem_dist = sem_dist / (sem_dist.sum() + self.eps)
        logits = self.graph.flow(sem_dist)
        probs = logits / (logits.sum() + self.eps)
        meta = {"num_edges": int(self.graph.num_edges)}
        return BridgeOutput(freq_logits=logits, freq_probs=probs, meta=meta)
