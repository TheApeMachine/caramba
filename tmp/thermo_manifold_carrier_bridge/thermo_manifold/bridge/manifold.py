from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..core.scatter import scatter_sum, segment_softmax


@dataclass
class BridgeOutput:
    """Spectral readout from the bridge."""

    spec_logits: torch.Tensor
    spec_probs: torch.Tensor
    meta: Dict[str, Any]


class BridgeManifold:
    """Vector-to-vector transduction via emergent carriers.

    This replaces the old ID->ID bipartite graph with a carrier population that couples
    a semantic vector space (R^D) to a spectral coordinate space (R).

    Design goals:
    - No dense semantic->spectral lookup table.
    - No backprop.
    - No Python-side per-edge loops.
    - Generalization via continuous semantic vectors.
    """

    def __init__(
        self,
        *,
        sem_dim: int,
        spec_bins: torch.Tensor,
        dt: float,
        device: torch.device,
        num_carriers: Optional[int] = None,
        eps: float = 1e-8,
    ):
        self.device = device
        self.dt = float(dt)
        self.eps = float(eps)

        self.sem_dim = int(sem_dim)

        bins = spec_bins.to(device=device, dtype=torch.float32).flatten()
        if bins.numel() == 0:
            raise ValueError("spec_bins must be non-empty")
        # Keep bins sorted to enable event-horizon neighbor search.
        self.spec_bins, self._bin_order = torch.sort(bins)
        self.num_bins = int(self.spec_bins.numel())

        if num_carriers is None:
            # Emergent-ish default: carriers scale sublinearly with output resolution.
            num_carriers = max(1, int(round(math.sqrt(self.num_bins))))
        self.num_carriers = int(num_carriers)

        # Carrier state.
        self.sem_pos = torch.randn(self.num_carriers, self.sem_dim, device=device, dtype=torch.float32)
        self.sem_pos = self.sem_pos / (self.sem_pos.norm(dim=1, keepdim=True) + self.eps)

        # Initialize spectral positions by sampling from the output bin range.
        lo = float(self.spec_bins.min().item())
        hi = float(self.spec_bins.max().item())
        if math.isclose(lo, hi):
            self.spec_pos = torch.full((self.num_carriers,), lo, device=device, dtype=torch.float32)
        else:
            u = torch.rand(self.num_carriers, device=device, dtype=torch.float32)
            self.spec_pos = lo + (hi - lo) * u

        self.energy = torch.zeros(self.num_carriers, device=device, dtype=torch.float32)
        self.heat = torch.zeros(self.num_carriers, device=device, dtype=torch.float32)

        # Homeostasis baseline.
        self._energy_baseline: Optional[torch.Tensor] = None

    # ----------------------------
    # Homeostasis
    # ----------------------------

    def _ratio(self, total: torch.Tensor) -> torch.Tensor:
        """Return total / baseline with an emergent baseline timescale."""
        dt = self.dt
        eps = self.eps
        total = total.to(torch.float32)
        if self._energy_baseline is None:
            self._energy_baseline = total.detach().clone()
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        base = self._energy_baseline
        # Baseline adapts slower when energy scale is larger.
        alpha = dt / (dt + float(base.abs().item()) + eps)
        base_new = base * (1.0 - alpha) + total.detach() * alpha
        self._energy_baseline = base_new
        return total / (base_new + eps)

    # ----------------------------
    # Core coupling primitives
    # ----------------------------

    def _semantic_bind(self, sem_pos: torch.Tensor, sem_energy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (assign[N,K], mass_in[K]) from semantic particles to carriers."""
        eps = self.eps

        sem_pos = sem_pos.to(device=self.device, dtype=torch.float32)
        sem_energy = sem_energy.to(device=self.device, dtype=torch.float32).flatten()
        if sem_pos.ndim == 1:
            sem_pos = sem_pos.view(1, -1)
        if sem_pos.numel() == 0:
            assign = torch.empty(0, self.num_carriers, device=self.device, dtype=torch.float32)
            mass_in = torch.zeros(self.num_carriers, device=self.device, dtype=torch.float32)
            return assign, mass_in

        # Distances: [N,K]
        # Using squared L2 keeps things smooth and avoids an extra sqrt.
        diff = sem_pos.unsqueeze(1) - self.sem_pos.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=2)
        d_scale = torch.sqrt(dist2.mean() + eps)

        # Carrier heat broadens binding (hotter => flatter).
        heat_level = self.heat.abs().mean()
        sharpness = (1.0 / (d_scale + eps)) / (1.0 + heat_level)
        logits = -dist2 * sharpness

        # Softmax per semantic particle.
        seg = torch.arange(int(sem_pos.shape[0]), device=self.device, dtype=torch.long)
        seg = seg.repeat_interleave(self.num_carriers)
        logits_e = logits.reshape(-1)
        assign_e = segment_softmax(logits_e, seg, int(sem_pos.shape[0]), eps=eps)
        assign = assign_e.view(int(sem_pos.shape[0]), self.num_carriers)

        mass_in = assign.T @ sem_energy
        return assign, mass_in

    def _spectral_bind(self, spec_pos: torch.Tensor, spec_energy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (assign[T,K], mass_in[K]) from spectral particles to carriers."""
        eps = self.eps

        spec_pos = spec_pos.to(device=self.device, dtype=torch.float32).flatten()
        spec_energy = spec_energy.to(device=self.device, dtype=torch.float32).flatten()
        if spec_pos.numel() == 0:
            assign = torch.empty(0, self.num_carriers, device=self.device, dtype=torch.float32)
            mass_in = torch.zeros(self.num_carriers, device=self.device, dtype=torch.float32)
            return assign, mass_in

        # Distances: [T,K]
        dist = (spec_pos.unsqueeze(1) - self.spec_pos.unsqueeze(0)).abs()
        d_scale = dist.mean() + eps

        heat_level = self.heat.abs().mean()
        sharpness = (1.0 / d_scale) / (1.0 + heat_level)
        logits = -dist * sharpness

        seg = torch.arange(int(spec_pos.shape[0]), device=self.device, dtype=torch.long)
        seg = seg.repeat_interleave(self.num_carriers)
        logits_e = logits.reshape(-1)
        assign_e = segment_softmax(logits_e, seg, int(spec_pos.shape[0]), eps=eps)
        assign = assign_e.view(int(spec_pos.shape[0]), self.num_carriers)

        mass_in = assign.T @ spec_energy
        return assign, mass_in

    # ----------------------------
    # Learning / adaptation (no backprop)
    # ----------------------------

    def observe(
        self,
        *,
        sem_pos: torch.Tensor,
        sem_energy: Optional[torch.Tensor] = None,
        spec_pos: torch.Tensor,
        spec_energy: Optional[torch.Tensor] = None,
    ) -> None:
        """Provide concurrent semantic + spectral evidence.

        This is the only "training" interface: the bridge updates its carriers
        based on co-activation, using purely local statistics.
        """

        eps = self.eps
        dt = self.dt

        if sem_pos.ndim == 1:
            sem_pos = sem_pos.view(1, -1)
        if sem_energy is None:
            sem_energy = torch.ones(int(sem_pos.shape[0]), device=self.device, dtype=torch.float32)
        else:
            sem_energy = sem_energy.to(device=self.device, dtype=torch.float32).flatten()

        spec_pos = spec_pos.to(device=self.device, dtype=torch.float32).flatten()
        if spec_energy is None:
            spec_energy = torch.ones(int(spec_pos.shape[0]), device=self.device, dtype=torch.float32)
        else:
            spec_energy = spec_energy.to(device=self.device, dtype=torch.float32).flatten()

        # Normalize input energies so update magnitudes are scale-free.
        sem_energy = sem_energy.clamp(min=0.0)
        sem_energy = sem_energy / (sem_energy.sum() + eps)
        spec_energy = spec_energy.clamp(min=0.0)
        spec_energy = spec_energy / (spec_energy.sum() + eps)

        sem_assign, sem_mass = self._semantic_bind(sem_pos, sem_energy)
        spec_assign, spec_mass = self._spectral_bind(spec_pos, spec_energy)

        # Carrier targets in semantic space.
        if sem_assign.numel() > 0:
            sem_weighted = sem_assign.T @ (sem_pos * sem_energy.unsqueeze(1))  # [K,D]
            sem_target = sem_weighted / (sem_mass.unsqueeze(1) + eps)
        else:
            sem_target = self.sem_pos

        # Carrier targets in spectral space.
        if spec_assign.numel() > 0:
            spec_weighted = spec_assign.T @ (spec_pos * spec_energy)  # [K]
            spec_target = spec_weighted / (spec_mass + eps)
        else:
            spec_target = self.spec_pos

        # Co-activation signal (Hebbian, but at carrier level).
        sem_scale = sem_mass.mean() + eps
        spec_scale = spec_mass.mean() + eps
        sem_n = sem_mass / sem_scale
        spec_n = spec_mass / spec_scale
        co = sem_n * spec_n

        # Incoherence generates heat.
        mismatch = (sem_n - spec_n).abs()

        total = sem_energy.sum() + spec_energy.sum() + self.energy.abs().sum()
        ratio = self._ratio(total)

        # Position updates: relax toward current evidence.
        self.sem_pos = self.sem_pos + dt * (sem_target - self.sem_pos)
        self.sem_pos = self.sem_pos / (self.sem_pos.norm(dim=1, keepdim=True) + eps)

        self.spec_pos = self.spec_pos + dt * (spec_target - self.spec_pos)

        # Energy + heat: homeostatic metabolism.
        e_scale = self.energy.abs().mean() + eps
        intake = co
        cost = ratio * self.energy / e_scale
        self.energy = (self.energy + dt * (intake - cost)).clamp(min=0.0)

        h_scale = self.heat.abs().mean() + eps
        self.heat = (self.heat + dt * (mismatch - ratio * self.heat / h_scale)).clamp(min=0.0)

    # ----------------------------
    # Readout
    # ----------------------------

    def forward(self, sem_pos: torch.Tensor, sem_energy: Optional[torch.Tensor] = None) -> BridgeOutput:
        """Project semantic particle(s) into a spectral distribution over spec_bins."""

        eps = self.eps

        if sem_pos.ndim == 1:
            sem_pos = sem_pos.view(1, -1)
        sem_pos = sem_pos.to(device=self.device, dtype=torch.float32)
        if sem_energy is None:
            sem_energy = torch.ones(int(sem_pos.shape[0]), device=self.device, dtype=torch.float32)
        else:
            sem_energy = sem_energy.to(device=self.device, dtype=torch.float32).flatten()
        sem_energy = sem_energy.clamp(min=0.0)
        sem_energy = sem_energy / (sem_energy.sum() + eps)

        # Carrier activations from semantic input.
        sem_assign, sem_mass = self._semantic_bind(sem_pos, sem_energy)
        carrier_act = sem_mass * (self.energy / (self.energy.mean() + eps))
        carrier_act = carrier_act.clamp(min=0.0)

        if carrier_act.numel() == 0 or float(carrier_act.sum().item()) <= float(eps):
            logits = torch.zeros(self.num_bins, device=self.device, dtype=torch.float32)
            probs = logits
            meta = {"carriers": int(self.num_carriers), "active": 0}
            return BridgeOutput(spec_logits=logits, spec_probs=probs, meta=meta)

        # Event-horizon projection: each carrier distributes mass to its left/right bin neighbors.
        bins = self.spec_bins
        m = self.num_bins
        a_sorted = bins
        p = self.spec_pos
        ins = torch.searchsorted(a_sorted, p)
        left = (ins - 1).clamp(min=0, max=m - 1)
        right = ins.clamp(min=0, max=m - 1)
        src = torch.arange(self.num_carriers, device=self.device, dtype=torch.long)
        src2 = torch.cat([src, src], dim=0)
        dst2 = torch.cat([left, right], dim=0)

        # Remove duplicate edges where left==right.
        keep = torch.ones(src2.numel(), device=self.device, dtype=torch.bool)
        dup = left == right
        keep[self.num_carriers :][dup] = False
        src_e = src2[keep]
        dst_e = dst2[keep]

        # Distances on the spectral axis.
        d = (p[src_e] - a_sorted[dst_e]).abs()
        d_scale = d.mean() + eps
        heat_level = self.heat.abs().mean()
        sharpness = (1.0 / d_scale) / (1.0 + heat_level)
        logits_e = -d * sharpness

        # Softmax within each carrier's outgoing neighborhood.
        w_e = segment_softmax(logits_e, src_e, self.num_carriers, eps=eps)

        # Scatter to bins.
        contrib = carrier_act[src_e] * w_e
        logits = scatter_sum(contrib, dst_e, m)
        probs = logits / (logits.sum() + eps)

        meta = {
            "carriers": int(self.num_carriers),
            "active": int(torch.count_nonzero(carrier_act).item()),
            "energy_mean": float(self.energy.mean().item()),
            "heat_mean": float(self.heat.mean().item()),
        }
        return BridgeOutput(spec_logits=logits, spec_probs=probs, meta=meta)
