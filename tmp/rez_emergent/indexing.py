
"""
rez_emergent.indexing

Locality tools to create a "semantic event horizon" without brute-force all-to-all distance
computations.

Design constraints:
- No backprop
- No learned hyperparameters
- Derive index sizing from population scale (V) rather than fixed magic constants
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch


DTYPE_REAL = torch.float32


def _ceil_log2(n: int) -> int:
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(float(n))))


def _normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _pack_bits_to_int64(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: [N, B] bool
    returns: [N] int64 where bit i is bits[:, i]
    Assumes B <= 62 (safe for shifting inside int64).
    """
    if bits.dtype != torch.bool:
        raise ValueError("bits must be bool")
    N, B = bits.shape
    if B <= 0:
        return torch.zeros((N,), dtype=torch.int64, device=bits.device)
    if B > 62:
        raise ValueError("Too many bits to pack into int64 safely.")
    codes = torch.zeros((N,), dtype=torch.int64, device=bits.device)
    for i in range(B):
        codes |= (bits[:, i].to(torch.int64) << i)
    return codes


@dataclass
class LSHTable:
    proj: torch.Tensor              # [B, D]
    uniq_codes: torch.Tensor        # [U]
    starts: torch.Tensor            # [U]
    counts: torch.Tensor            # [U]
    sorted_perm: torch.Tensor       # [N]
    sorted_codes: torch.Tensor      # [N]


class LSHIndex:
    """
    Multi-table random-hyperplane LSH for cosine similarity.

    - Build once for (approximately) static attractor embeddings.
    - Query returns a candidate set; caller computes exact similarities on candidates.

    No tuned knobs:
      total_bits  = ceil(log2(N+1))
      n_tables    = ceil(sqrt(total_bits))
      bits/table  = ceil(total_bits / n_tables)
    """

    def __init__(self, embeddings: torch.Tensor, eps: float = 1e-8, seed: Optional[int] = None):
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be [N, D]")
        self.eps = float(eps)
        self.device = embeddings.device
        self.embeddings = _normalize_rows(embeddings.to(dtype=DTYPE_REAL), self.eps)
        self.N = int(self.embeddings.shape[0])
        self.D = int(self.embeddings.shape[1])

        total_bits = _ceil_log2(self.N + 1)
        n_tables = int(math.ceil(math.sqrt(float(total_bits))))
        bits_per_table = int(math.ceil(float(total_bits) / float(n_tables)))

        # Store derived sizing for transparency/debug
        self.total_bits = total_bits
        self.n_tables = n_tables
        self.bits_per_table = bits_per_table

        # Deterministic seed if provided; otherwise uses PyTorch RNG state.
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(int(seed))
        else:
            g = None

        self.tables: list[LSHTable] = []
        for _ in range(n_tables):
            proj = torch.randn((bits_per_table, self.D), device=self.device, dtype=DTYPE_REAL, generator=g)
            self.tables.append(self._build_table(proj))

    def _codes_for(self, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        x = _normalize_rows(x.to(device=self.device, dtype=DTYPE_REAL), self.eps)
        # [N, B]
        bits = (x @ proj.t()) > 0
        return _pack_bits_to_int64(bits)

    def _build_table(self, proj: torch.Tensor) -> LSHTable:
        codes = self._codes_for(self.embeddings, proj)  # [N]
        sorted_codes, perm = torch.sort(codes)
        uniq, counts = torch.unique_consecutive(sorted_codes, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros((1,), device=self.device, dtype=torch.int64), counts[:-1]]), dim=0)
        return LSHTable(
            proj=proj,
            uniq_codes=uniq,
            starts=starts,
            counts=counts,
            sorted_perm=perm,
            sorted_codes=sorted_codes,
        )

    def candidates(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [D] or [1, D]
        Returns unique candidate indices [C] (int64) gathered from buckets across tables.
        """
        if q.dim() == 1:
            qv = q.unsqueeze(0)
        elif q.dim() == 2 and q.shape[0] == 1:
            qv = q
        else:
            raise ValueError("q must be [D] or [1,D]. Batch queries are intentionally not supported here.")

        all_idx = []
        for tab in self.tables:
            code_q = self._codes_for(qv, tab.proj)[0]  # scalar
            # find in uniq_codes
            pos = torch.searchsorted(tab.uniq_codes, code_q)
            hit = (pos < tab.uniq_codes.numel()) and (tab.uniq_codes[pos] == code_q)
            if hit:
                start = int(tab.starts[pos].item())
                count = int(tab.counts[pos].item())
                all_idx.append(tab.sorted_perm[start : start + count])
            else:
                # fallback: take nearest existing bucket by ordering
                if tab.uniq_codes.numel() > 0:
                    pos_clamped = int(torch.clamp(pos, 0, tab.uniq_codes.numel() - 1).item())
                    start = int(tab.starts[pos_clamped].item())
                    count = int(tab.counts[pos_clamped].item())
                    all_idx.append(tab.sorted_perm[start : start + count])

        if not all_idx:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        cat = torch.cat(all_idx, dim=0)
        return torch.unique(cat)

    def topk(self, q: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (idx, sims) for top-k cosine sims among candidates.
        If candidates < k, returns all candidates.
        """
        if k <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device), torch.empty((0,), dtype=DTYPE_REAL, device=self.device)

        cands = self.candidates(q)
        if cands.numel() == 0:
            return cands, torch.empty((0,), dtype=DTYPE_REAL, device=self.device)

        qn = q.to(device=self.device, dtype=DTYPE_REAL)
        qn = qn / (qn.norm() + self.eps)

        sims = (self.embeddings.index_select(0, cands) @ qn)  # [C]
        kk = min(int(k), int(sims.numel()))
        vals, rel = torch.topk(sims, k=kk, largest=True, sorted=True)
        idx = cands.index_select(0, rel)
        return idx, vals
