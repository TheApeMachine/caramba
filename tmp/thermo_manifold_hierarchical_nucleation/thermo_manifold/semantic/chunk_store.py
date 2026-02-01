from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class LookupResult:
    idx: torch.Tensor
    exists: torch.Tensor
    key: torch.Tensor


class ChunkStore:
    """Dynamic store of higher-order "slow" particles (chunks).

    Key design constraint: chunk indices must remain stable once assigned, because
    other structures (e.g., chunk->token bond graphs) reference them.

    We therefore maintain:
    - An append-only set of chunk state tensors indexed by chunk_id (0..C-1)
    - A separate sorted key->chunk_id mapping for fast GPU lookups.

    Chunk "keys" are order-grams encoded in base vocab_size.
    """

    def __init__(
        self,
        *,
        order: int,
        vocab_size: int,
        sem_dim: int,
        device: torch.device,
        eps: float,
    ):
        if order < 2:
            raise ValueError('order must be >= 2')
        self.order = int(order)
        self.vocab_size = int(vocab_size)
        self.sem_dim = int(sem_dim)
        self.device = device
        self.eps = float(eps)

        # Stable chunk state (append-only)
        self.seq = torch.empty(0, self.order, device=device, dtype=torch.long)
        self.position = torch.empty(0, self.sem_dim, device=device, dtype=torch.float32)
        self.energy = torch.empty(0, device=device, dtype=torch.float32)
        self.excitation = torch.empty(0, device=device, dtype=torch.float32)
        self.heat = torch.empty(0, device=device, dtype=torch.float32)

        # Sorted key->chunk_id mapping for O(log C) lookup.
        self.key_sorted = torch.empty(0, device=device, dtype=torch.long)
        self.idx_sorted = torch.empty(0, device=device, dtype=torch.long)

        # Baseline for binding energies (homeostatic scale).
        self._binding_baseline: Optional[torch.Tensor] = None

    @property
    def num_chunks(self) -> int:
        return int(self.energy.numel())

    # ----------------------------
    # Keying / lookup
    # ----------------------------

    def _make_key(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq.to(device=self.device, dtype=torch.long)
        if seq.ndim != 2 or seq.shape[1] != self.order:
            raise ValueError(f'seq must have shape [N,{self.order}]')
        key = seq[:, 0]
        base = self.vocab_size
        for i in range(1, self.order):
            key = key * base + seq[:, i]
        return key

    def lookup(self, seq: torch.Tensor) -> LookupResult:
        """Lookup chunk_id(s) for a batch of sequences."""

        key = self._make_key(seq)
        if self.key_sorted.numel() == 0:
            idx = torch.full_like(key, -1, dtype=torch.long, device=self.device)
            exists = torch.zeros_like(key, dtype=torch.bool, device=self.device)
            return LookupResult(idx=idx, exists=exists, key=key)

        pos = torch.searchsorted(self.key_sorted, key)
        in_range = pos < self.key_sorted.numel()
        pos_safe = pos.clamp(max=max(int(self.key_sorted.numel()) - 1, 0))
        hit = self.key_sorted[pos_safe] == key
        exists = in_range & hit

        idx = torch.full_like(key, -1, dtype=torch.long, device=self.device)
        if bool(exists.any()):
            idx[exists] = self.idx_sorted[pos_safe[exists]]
        return LookupResult(idx=idx, exists=exists, key=key)

    # ----------------------------
    # Homeostatic baseline
    # ----------------------------

    def update_binding_baseline(self, binding: torch.Tensor, *, dt: float) -> torch.Tensor:
        """Update and return the binding baseline (scalar tensor)."""

        eps = self.eps
        b = binding.to(device=self.device, dtype=torch.float32).mean()
        if self._binding_baseline is None:
            self._binding_baseline = b.detach().clone()
            return self._binding_baseline

        base = self._binding_baseline
        alpha = float(dt) / (float(dt) + float(base.abs().item()) + eps)
        base_new = base * (1.0 - alpha) + b.detach() * alpha
        self._binding_baseline = base_new
        return self._binding_baseline

    @property
    def binding_baseline(self) -> torch.Tensor:
        if self._binding_baseline is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return self._binding_baseline

    # ----------------------------
    # Insertion / reinforcement
    # ----------------------------

    def add_or_reinforce(self, seq: torch.Tensor, mass: torch.Tensor, *, word_pos: torch.Tensor) -> LookupResult:
        """Add new chunks or reinforce existing ones.

        seq: [N,order]
        mass: scalar or [N]
        word_pos: [V,D] token embeddings (used to place new chunks)

        Returns a LookupResult aligned with the input seq rows.
        """

        eps = self.eps

        seq = seq.to(device=self.device, dtype=torch.long)
        if seq.ndim == 1:
            seq = seq.view(1, -1)
        if seq.ndim != 2 or seq.shape[1] != self.order:
            raise ValueError(f'seq must have shape [N,{self.order}]')

        mass = mass.to(device=self.device, dtype=torch.float32)
        if mass.numel() == 1:
            mass = mass.expand(int(seq.shape[0]))
        else:
            mass = mass.flatten()
            if mass.shape[0] != seq.shape[0]:
                raise ValueError('mass must be scalar or have shape [N]')

        # Filter: only positive mass contributes to structural reinforcement.
        pos_mask = mass > 0
        if not bool(pos_mask.any()):
            return self.lookup(seq)

        seq_p = seq[pos_mask]
        mass_p = mass[pos_mask]

        key_new = self._make_key(seq_p)

        # Coalesce duplicates in the new batch.
        order = torch.argsort(key_new)
        key_new = key_new[order]
        seq_p = seq_p[order]
        mass_p = mass_p[order]

        key_u, inv = torch.unique_consecutive(key_new, return_inverse=True)
        if key_u.numel() != key_new.numel():
            mass_u = torch.zeros(int(key_u.numel()), device=self.device, dtype=torch.float32)
            mass_u.index_add_(0, inv, mass_p)
            # keep the first representative sequence for each key
            first = torch.zeros(int(key_u.numel()), device=self.device, dtype=torch.long)
            first.scatter_(0, inv, torch.arange(int(inv.numel()), device=self.device, dtype=torch.long))
            seq_u = seq_p[first]
        else:
            mass_u = mass_p
            seq_u = seq_p

        # Lookup existing
        if self.key_sorted.numel() == 0:
            exists = torch.zeros(int(key_u.numel()), device=self.device, dtype=torch.bool)
            idx_existing = torch.full((int(key_u.numel()),), -1, device=self.device, dtype=torch.long)
            pos = torch.zeros(int(key_u.numel()), device=self.device, dtype=torch.long)
        else:
            pos = torch.searchsorted(self.key_sorted, key_u)
            in_range = pos < self.key_sorted.numel()
            pos_safe = pos.clamp(max=max(int(self.key_sorted.numel()) - 1, 0))
            hit = self.key_sorted[pos_safe] == key_u
            exists = in_range & hit
            idx_existing = torch.full((int(key_u.numel()),), -1, device=self.device, dtype=torch.long)
            if bool(exists.any()):
                idx_existing[exists] = self.idx_sorted[pos_safe[exists]]

        # Reinforce existing chunks.
        if bool(exists.any()):
            idx_e = idx_existing[exists]
            m_e = mass_u[exists]
            self.energy[idx_e] = self.energy[idx_e] + m_e
            self.excitation[idx_e] = self.excitation[idx_e] + m_e

        # Insert new chunks.
        add_mask = ~exists
        if bool(add_mask.any()):
            key_add = key_u[add_mask]
            seq_add = seq_u[add_mask]
            mass_add = mass_u[add_mask]

            # Append stable chunk state.
            start = self.num_chunks
            add_n = int(key_add.numel())
            idx_add = torch.arange(start, start + add_n, device=self.device, dtype=torch.long)

            # Place new chunks at the normalized sum of constituent token embeddings.
            wp = word_pos.to(device=self.device, dtype=torch.float32)
            pos_add = wp[seq_add].sum(dim=1)
            pos_add = pos_add / (pos_add.norm(dim=1, keepdim=True) + eps)

            self.seq = torch.cat([self.seq, seq_add], dim=0)
            self.position = torch.cat([self.position, pos_add], dim=0)
            self.energy = torch.cat([self.energy, mass_add], dim=0)
            self.excitation = torch.cat([self.excitation, mass_add], dim=0)
            self.heat = torch.cat([self.heat, torch.zeros(add_n, device=self.device, dtype=torch.float32)], dim=0)

            # Update sorted key mapping (merge, no full sort).
            if self.key_sorted.numel() == 0:
                self.key_sorted = key_add
                self.idx_sorted = idx_add
            else:
                # key_add is already sorted because key_u is sorted. Ensure.
                key_add, order2 = torch.sort(key_add)
                idx_add = idx_add[order2]
                pos_add = torch.searchsorted(self.key_sorted, key_add)

                old_n = int(self.key_sorted.numel())
                add_n = int(key_add.numel())
                merged_n = old_n + add_n

                new_pos = pos_add + torch.arange(add_n, device=self.device, dtype=torch.long)
                old_i = torch.arange(old_n, device=self.device, dtype=torch.long)
                shift = torch.searchsorted(pos_add, old_i, right=True)
                old_pos = old_i + shift

                key_m = torch.empty(merged_n, device=self.device, dtype=torch.long)
                idx_m = torch.empty(merged_n, device=self.device, dtype=torch.long)

                key_m[old_pos] = self.key_sorted
                idx_m[old_pos] = self.idx_sorted

                key_m[new_pos] = key_add
                idx_m[new_pos] = idx_add

                self.key_sorted = key_m
                self.idx_sorted = idx_m

        return self.lookup(seq)

    # ----------------------------
    # Dynamics
    # ----------------------------

    def decay(self, *, ratio: torch.Tensor, dt: float) -> None:
        """Homeostatic decay for chunk reservoirs (no reinforcement)."""

        if self.num_chunks == 0:
            return

        eps = self.eps
        dt = float(dt)
        ratio = ratio.to(device=self.device, dtype=torch.float32)

        e = self.energy
        x = self.excitation
        h = self.heat

        e_scale = e.abs().mean() + eps
        x_scale = x.abs().mean() + eps
        h_scale = h.abs().mean() + eps

        e_decay = torch.exp(-dt * ratio / e_scale)
        x_decay = torch.exp(-dt * ratio / x_scale)
        h_decay = torch.exp(-dt * ratio / h_scale)

        self.energy = (e * e_decay).clamp(min=0.0)
        self.excitation = (x * x_decay).clamp(min=0.0)
        self.heat = (h * h_decay).clamp(min=0.0)

    def distribution(self) -> torch.Tensor:
        """Return a normalized distribution over chunks based on excitation."""

        if self.num_chunks == 0:
            return torch.zeros(0, device=self.device, dtype=torch.float32)

        eps = self.eps
        x = self.excitation.clamp(min=0.0)
        s = x.sum()
        if float(s.item()) <= eps:
            return torch.zeros_like(x)
        return x / (s + eps)
