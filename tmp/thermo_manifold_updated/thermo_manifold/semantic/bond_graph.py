from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class BondBatch:
    src: torch.Tensor
    dst: torch.Tensor
    w: torch.Tensor
    trace: torch.Tensor


class SparseBondGraph:
    """Sparse directed bond graph with per-edge mass and eligibility trace.

    Storage:
    - Edges are kept in contiguous torch tensors for fast scatter ops.
    - Python-side index maps are used only for edge insertion / lookup.
    """

    def __init__(self, num_nodes: int, *, device: torch.device, dtype: torch.dtype, eps: float):
        self.num_nodes = int(num_nodes)
        self.device = device
        self.dtype = dtype
        self.eps = float(eps)

        self.src = torch.empty(0, dtype=torch.long, device=device)
        self.dst = torch.empty(0, dtype=torch.long, device=device)
        self.w = torch.empty(0, dtype=dtype, device=device)
        self.trace = torch.empty(0, dtype=dtype, device=device)

        self._edge_index: Dict[Tuple[int, int], int] = {}
        self._out_edges: List[List[int]] = [[] for _ in range(self.num_nodes)]

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def edges_from(self, src_id: int) -> List[int]:
        return self._out_edges[int(src_id)]

    def add_edge(self, src_id: int, dst_id: int, init_mass: torch.Tensor) -> int:
        """Add (or reinforce) an edge and return its edge index."""
        key = (int(src_id), int(dst_id))
        if key in self._edge_index:
            ei = self._edge_index[key]
            self.w[ei] = self.w[ei] + init_mass.to(self.w.dtype)
            return ei

        ei = self.num_edges
        self._edge_index[key] = ei
        self._out_edges[key[0]].append(ei)

        self.src = torch.cat([self.src, torch.tensor([key[0]], device=self.device, dtype=torch.long)], dim=0)
        self.dst = torch.cat([self.dst, torch.tensor([key[1]], device=self.device, dtype=torch.long)], dim=0)
        self.w = torch.cat([self.w, init_mass.to(self.dtype).view(1)], dim=0)
        self.trace = torch.cat([self.trace, torch.zeros(1, device=self.device, dtype=self.dtype)], dim=0)
        return ei

    def add_path(self, ids: torch.Tensor, mass: torch.Tensor) -> None:
        """Add edges for a sequence of ids: ids[t] -> ids[t+1]."""
        if ids.numel() < 2:
            return
        for a, b in zip(ids[:-1].tolist(), ids[1:].tolist()):
            self.add_edge(int(a), int(b), mass)

    def out_sum(self, src_ids: torch.Tensor) -> torch.Tensor:
        """Outgoing mass per src (for provided src ids)."""
        out = torch.zeros(int(src_ids.numel()), device=self.device, dtype=self.dtype)
        for i, s in enumerate(src_ids.tolist()):
            eidx = self._out_edges[int(s)]
            if eidx:
                out[i] = self.w[torch.tensor(eidx, device=self.device)].sum()
        return out

    def flow_from_distribution(self, dist: torch.Tensor) -> torch.Tensor:
        """One-step flow of a distribution through the graph.

        dist: [V] nonnegative mass over sources
        returns: [V] mass over destinations
        """
        if self.num_edges == 0:
            return torch.zeros_like(dist)

        # Normalize outgoing weights per source.
        out_sum = torch.zeros(self.num_nodes, device=self.device, dtype=self.dtype)
        out_sum.index_add_(0, self.src, self.w)
        w_norm = self.w / (out_sum[self.src] + self.eps)

        contrib = dist[self.src] * w_norm
        out = torch.zeros_like(dist)
        out.index_add_(0, self.dst, contrib)
        return out

    def batch_edges(self, src_ids: torch.Tensor) -> Optional[BondBatch]:
        """Collect all outgoing edges for the provided sources."""
        if src_ids.numel() == 0 or self.num_edges == 0:
            return None
        idx: List[int] = []
        for s in src_ids.tolist():
            idx.extend(self._out_edges[int(s)])
        if not idx:
            return None
        eidx = torch.tensor(idx, device=self.device, dtype=torch.long)
        return BondBatch(src=self.src[eidx], dst=self.dst[eidx], w=self.w[eidx], trace=self.trace[eidx])

    def update_edges(self, eidx: torch.Tensor, w_new: torch.Tensor, trace_new: torch.Tensor) -> None:
        self.w[eidx] = w_new
        self.trace[eidx] = trace_new

    def prune_by_src_mean(self, src_ids: torch.Tensor) -> None:
        """Prune weak edges using a per-source mean threshold (no fixed constants).

        For each source, keep edges with w >= mean(w_out(src)).
        """
        for s in src_ids.tolist():
            eidx = self._out_edges[int(s)]
            if not eidx:
                continue
            w = self.w[torch.tensor(eidx, device=self.device)]
            mean = w.mean()
            keep = w >= mean
            if keep.all():
                continue
            # Zero-out pruned edges (lazy pruning).
            pruned_idx = torch.tensor([eidx[i] for i, k in enumerate(keep.tolist()) if not k], device=self.device)
            if pruned_idx.numel() > 0:
                self.w[pruned_idx] = torch.zeros_like(self.w[pruned_idx])
                self.trace[pruned_idx] = torch.zeros_like(self.trace[pruned_idx])
