
"""
rez_emergent.physics

Domain-agnostic thermodynamic dynamics with:
- emergent Boltzmann resonance
- optional local interaction horizon for 1D manifolds (avoid O(N*M))
- homeostatic normalization (self-tuning thermostat without fixed targets)

This file is a forward-direction replacement for the original physics.py,
which computed full pairwise distances every step. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import math
import torch

DTYPE_REAL = torch.float32
TAU = 2.0 * math.pi


@dataclass
class PhysicsConfig:
    dt: float = 0.05
    eps: float = 1e-8

    # Hard caps are engineering constraints, not tuned "hyperparameters".
    max_particles: int = 4096

    # If True, 1D manifolds use a local neighbor window to avoid O(N*M).
    local_1d_horizon: bool = True



@dataclass
class OutputState:
    """Unified output container for semantic/audio domains."""
    logits: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    token_index: Optional[int] = None
    token: Optional[str] = None
    audio_particles: Optional[torch.Tensor] = None
    audio_targets: Optional[torch.Tensor] = None
    meta: Optional[dict[str, Any]] = None

def _row_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)


def _normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / _row_norm(x, eps).unsqueeze(-1)


class Homeostat:
    """
    A self-normalizing thermostat.

    Tracks a baseline of total energy and returns a dimensionless ratio:
        ratio = total / baseline

    No target thresholds; baseline is learned from the system itself.
    """

    def __init__(self):
        self.baseline: Optional[torch.Tensor] = None

    def update(self, total: torch.Tensor, dt: float, eps: float) -> torch.Tensor:
        total = total.to(dtype=DTYPE_REAL)
        if self.baseline is None:
            self.baseline = total.detach()
            return torch.ones((), dtype=DTYPE_REAL, device=total.device)

        base = self.baseline
        # Emergent smoothing: the larger the baseline energy, the faster baseline can track changes.
        scale = base.abs().clamp_min(eps)
        tau = 1.0 / scale  # self-scaled timescale
        alpha = dt / (tau + dt)
        self.baseline = base * (1.0 - alpha) + total.detach() * alpha
        return total / (self.baseline.abs().clamp_min(eps))


class ThermodynamicEngine:
    """
    Particles and attractors evolve via:
      - resonance weights (Boltzmann-like) derived from distances and emergent temperature
      - drift toward weighted targets
      - Langevin noise scaled by temperature
      - thermodynamic bookkeeping (energy/heat diffusion)
      - homeostatic scaling of decay (thermostat)

    IMPORTANT scalability note:
      The original implementation computed full [N,M] distances every step. 
      Here, if positions are 1D and local_1d_horizon=True, we restrict interactions to a
      local neighbor window of size K≈sqrt(M), yielding O(N*K) per step.
    """

    def __init__(self, config: PhysicsConfig, device: torch.device):
        self.config = config
        self.device = device
        self.t = 0.0

        self.particles: dict[str, torch.Tensor] = {}
        self.attractors: dict[str, torch.Tensor] = {}

        self._homeostat = Homeostat()

    def _count(self, state: dict[str, torch.Tensor]) -> int:
        pos = state.get("position", None)
        return int(pos.shape[0]) if pos is not None else 0

    def _total_energy(self) -> torch.Tensor:
        total = torch.zeros((), dtype=DTYPE_REAL, device=self.device)
        for st in (self.particles, self.attractors):
            e = st.get("energy", None)
            if e is not None and e.numel() > 0:
                total = total + e.sum()
        return total

    def _neighbor_window_1d(self, p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Return neighbor indices [N,K] for 1D a (attractor positions).
        K is emergent: ceil(sqrt(M)).
        """
        M = int(a.shape[0])
        if M == 0:
            return torch.empty((int(p.shape[0]), 0), dtype=torch.int64, device=self.device)
        K = int(math.ceil(math.sqrt(float(M))))
        K = min(K, M)

        a_sorted, perm = torch.sort(a)
        # insertion positions in sorted array
        pos = torch.searchsorted(a_sorted, p)
        # center window on pos
        half = K // 2
        start = (pos - half).clamp(min=0, max=max(0, M - K))
        offsets = torch.arange(K, device=self.device, dtype=torch.int64).unsqueeze(0)
        win = start.to(torch.int64).unsqueeze(1) + offsets  # [N,K] in sorted index space
        idx = perm.index_select(0, win.reshape(-1)).reshape(win.shape)  # map to original indices
        return idx

    def _compute_weights_dense(
        self, particle_positions: torch.Tensor, attractor_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = float(self.config.eps)

        if particle_positions.dim() == 1:
            dists = torch.abs(particle_positions.unsqueeze(1) - attractor_positions.unsqueeze(0))
        else:
            dists = _row_norm(particle_positions.unsqueeze(1) - attractor_positions.unsqueeze(0), eps)

        # Emergent distance scale
        scale = torch.sqrt(torch.mean(dists * dists)).clamp_min(eps)

        # Emergent temperature from particle heat
        heat = self.particles.get("heat", None)
        if heat is None or heat.numel() == 0:
            temp = scale
        else:
            h_scale = heat.mean().abs().clamp_min(eps)
            temp = scale * (1.0 + (heat.unsqueeze(1) / h_scale))

        logits = -dists / temp.clamp_min(eps)
        logits = logits - logits.max(dim=1, keepdim=True).values
        w = torch.exp(logits)

        # Attractor "mass" (energy) biases capture, analogous to stronger wells.
        a_energy = self.attractors.get("energy", None)
        if a_energy is not None and a_energy.numel() == attractor_positions.shape[0]:
            m = a_energy.clamp_min(0.0)
            m_scale = m.mean().clamp_min(eps)
            m = (m / m_scale).clamp_min(0.0)
            w = w * (m.unsqueeze(0) + eps)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        return w, dists

    def _compute_weights_sparse_1d(
        self, particle_positions: torch.Tensor, attractor_positions: torch.Tensor, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = float(self.config.eps)
        # Gather attractor positions for each particle
        a_sel = attractor_positions.index_select(0, idx.reshape(-1)).reshape(idx.shape)  # [N,K]
        dists = torch.abs(particle_positions.unsqueeze(1) - a_sel)

        scale = torch.sqrt(torch.mean(dists * dists)).clamp_min(eps)

        heat = self.particles.get("heat", None)
        if heat is None or heat.numel() == 0:
            temp = scale
        else:
            h_scale = heat.mean().abs().clamp_min(eps)
            temp = scale * (1.0 + (heat.unsqueeze(1) / h_scale))

        logits = -dists / temp.clamp_min(eps)
        logits = logits - logits.max(dim=1, keepdim=True).values
        w = torch.exp(logits)

        # Attractor mass bias on selected indices
        a_energy = self.attractors.get("energy", None)
        if a_energy is not None and a_energy.numel() == attractor_positions.shape[0]:
            m = a_energy.clamp_min(0.0)
            m_scale = m.mean().clamp_min(eps)
            m = (m / m_scale).clamp_min(0.0)
            m_sel = m.index_select(0, idx.reshape(-1)).reshape(idx.shape)
            w = w * (m_sel + eps)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        return w, dists

    def _scatter_add(self, out: torch.Tensor, idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        """
        out: [M]
        idx: [N,K] int64
        val: [N,K]
        """
        flat_i = idx.reshape(-1)
        flat_v = val.reshape(-1)
        return out.scatter_add(0, flat_i, flat_v)

    def update_thermodynamics_dense(
        self,
        weights: torch.Tensor,
        dists: torch.Tensor,
        prev_positions: torch.Tensor,
        new_positions: torch.Tensor,
        dt_eff: float,
    ) -> None:
        dt = float(dt_eff)
        eps = float(self.config.eps)

        N = int(weights.shape[0])
        M = int(weights.shape[1])

        # Ensure fields exist
        if "heat" not in self.particles:
            self.particles["heat"] = torch.zeros((N,), dtype=DTYPE_REAL, device=self.device)
        if "energy" not in self.particles:
            self.particles["energy"] = torch.ones((N,), dtype=DTYPE_REAL, device=self.device)

        if "heat" not in self.attractors:
            self.attractors["heat"] = torch.zeros((M,), dtype=DTYPE_REAL, device=self.device)
        if "energy" not in self.attractors:
            self.attractors["energy"] = torch.zeros((M,), dtype=DTYPE_REAL, device=self.device)

        # Activity (kinetic proxy)
        if prev_positions.dim() == 1:
            disp = torch.abs(new_positions - prev_positions)
        else:
            disp = _row_norm(new_positions - prev_positions, eps)

        disp_scale = torch.sqrt(torch.mean(disp * disp)).clamp_min(eps)
        activity = disp / disp_scale

        d_scale = torch.sqrt(torch.mean(dists * dists)).clamp_min(eps)
        mismatch = (dists / d_scale).mean(dim=1)

        heat_in = activity * mismatch
        self.particles["heat"] = self.particles["heat"] + dt * heat_in

        # Diffuse heat to attractors
        heat_to_attr = torch.matmul(weights.t(), heat_in)
        self.attractors["heat"] = self.attractors["heat"] + dt * heat_to_attr

        # Energy transfer to attractors
        p_energy = self.particles["energy"].clamp_min(0.0)
        energy_to_attr = torch.matmul(weights.t(), p_energy)

        # Homeostatic decay (self-scaled)
        E = self.attractors["energy"]
        E_scale = E.abs().mean().clamp_min(eps)
        leak = torch.exp(-dt / E_scale)
        self.attractors["energy"] = E * leak + (1.0 - leak) * energy_to_attr

        # Cool particle heat
        H = self.particles["heat"]
        H_scale = H.abs().mean().clamp_min(eps)
        cool = torch.exp(-dt / H_scale)
        self.particles["heat"] = H * cool

    def update_thermodynamics_sparse_1d(
        self,
        weights: torch.Tensor,      # [N,K]
        idx: torch.Tensor,          # [N,K] attractor indices
        dists: torch.Tensor,        # [N,K]
        prev_positions: torch.Tensor,
        new_positions: torch.Tensor,
        M: int,
        dt_eff: float,
    ) -> None:
        dt = float(dt_eff)
        eps = float(self.config.eps)
        N = int(weights.shape[0])

        # Ensure fields exist
        if "heat" not in self.particles:
            self.particles["heat"] = torch.zeros((N,), dtype=DTYPE_REAL, device=self.device)
        if "energy" not in self.particles:
            self.particles["energy"] = torch.ones((N,), dtype=DTYPE_REAL, device=self.device)

        if "heat" not in self.attractors:
            self.attractors["heat"] = torch.zeros((M,), dtype=DTYPE_REAL, device=self.device)
        if "energy" not in self.attractors:
            self.attractors["energy"] = torch.zeros((M,), dtype=DTYPE_REAL, device=self.device)

        disp = torch.abs(new_positions - prev_positions)
        disp_scale = torch.sqrt(torch.mean(disp * disp)).clamp_min(eps)
        activity = disp / disp_scale

        d_scale = torch.sqrt(torch.mean(dists * dists)).clamp_min(eps)
        mismatch = (dists / d_scale).mean(dim=1)

        heat_in = activity * mismatch
        self.particles["heat"] = self.particles["heat"] + dt * heat_in

        # Scatter heat to attractors
        heat_to_attr = torch.zeros((M,), dtype=DTYPE_REAL, device=self.device)
        heat_to_attr = self._scatter_add(heat_to_attr, idx, weights * heat_in.unsqueeze(1))
        self.attractors["heat"] = self.attractors["heat"] + dt * heat_to_attr

        # Scatter energy to attractors
        p_energy = self.particles["energy"].clamp_min(0.0)
        energy_to_attr = torch.zeros((M,), dtype=DTYPE_REAL, device=self.device)
        energy_to_attr = self._scatter_add(energy_to_attr, idx, weights * p_energy.unsqueeze(1))

        E = self.attractors["energy"]
        E_scale = E.abs().mean().clamp_min(eps)
        leak = torch.exp(-dt / E_scale)
        self.attractors["energy"] = E * leak + (1.0 - leak) * energy_to_attr

        # Cool particle heat
        H = self.particles["heat"]
        H_scale = H.abs().mean().clamp_min(eps)
        cool = torch.exp(-dt / H_scale)
        self.particles["heat"] = H * cool

    def step_physics(self) -> None:
        N = self._count(self.particles)
        M = self._count(self.attractors)
        dt = float(self.config.dt)
        eps = float(self.config.eps)

        if N == 0 or M == 0:
            self.t += dt
            return

        # Homeostatic ratio from total system energy (self-defined target)
        ratio = self._homeostat.update(self._total_energy(), dt=dt, eps=eps)
        dt_eff = float((dt * ratio).item())

        current = self.particles["position"]
        targets = self.attractors["position"]

        # Use local horizon only for 1D manifolds
        if (
            self.config.local_1d_horizon
            and current.dim() == 1
            and targets.dim() == 1
            and M > 0
        ):
            idx = self._neighbor_window_1d(current, targets)  # [N,K]
            weights, dists = self._compute_weights_sparse_1d(current, targets, idx)

            # Weighted target position: sum_k w * a_pos[idx]
            a_sel = targets.index_select(0, idx.reshape(-1)).reshape(idx.shape)
            weighted_target = torch.sum(weights * a_sel, dim=1)  # [N]

            drift = weighted_target - current

            # Temperature from particle heat (or distance scale)
            heat = self.particles.get("heat", None)
            if heat is None or heat.numel() == 0:
                T_vec = torch.full((N,), torch.sqrt(torch.mean(dists * dists)).clamp_min(eps).item(), device=self.device, dtype=DTYPE_REAL)
            else:
                h_scale = heat.abs().mean().clamp_min(eps)
                T_vec = (heat / h_scale).clamp_min(0.0) + 1.0

            noise = torch.randn_like(current) * torch.sqrt(2.0 * T_vec * dt_eff)
            prev = current
            new = current + drift * dt_eff + noise

            self.particles["position"] = new
            self.update_thermodynamics_sparse_1d(weights, idx, dists, prev, new, M=M, dt_eff=dt_eff)

        else:
            weights, dists = self._compute_weights_dense(current, targets)

            if current.dim() == 1:
                weighted_target = torch.matmul(weights, targets)  # [N]
            else:
                weighted_target = torch.matmul(weights, targets)  # [N,D]

            drift = weighted_target - current

            heat = self.particles.get("heat", None)
            if heat is None or heat.numel() == 0:
                T = torch.sqrt(torch.mean(dists * dists)).clamp_min(eps)
                T_vec = torch.full((N,), float(T.item()), dtype=DTYPE_REAL, device=self.device)
            else:
                h_scale = heat.abs().mean().clamp_min(eps)
                T_vec = (heat / h_scale).clamp_min(0.0) + 1.0

            if current.dim() == 1:
                noise = torch.randn_like(current) * torch.sqrt(2.0 * T_vec * dt_eff)
            else:
                noise = torch.randn_like(current) * torch.sqrt(2.0 * T_vec.unsqueeze(1) * dt_eff)

            prev = current
            new = current + drift * dt_eff + noise

            self.particles["position"] = new
            self.update_thermodynamics_dense(weights, dists, prev_positions=prev, new_positions=new, dt_eff=dt_eff)

        self.t += dt


class SpectralManifold(ThermodynamicEngine):
    """
    1D frequency manifold ("voice").

    Particles: frequencies
    Attractors: target frequencies (with optional energy masses)
    """

    def __init__(self, config: PhysicsConfig, device: torch.device):
        super().__init__(config, device)
        self.particles = {
            "position": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "phase": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "energy": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "heat": torch.empty((0,), dtype=DTYPE_REAL, device=device),
        }
        self.attractors = {
            "position": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "energy": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "heat": torch.empty((0,), dtype=DTYPE_REAL, device=device),
        }

    def seed_noise(self, n: int, f_min: float, f_max: float) -> None:
        eps = float(self.config.eps)
        n = int(n)
        if n <= 0:
            return
        freqs = (torch.rand((n,), device=self.device, dtype=DTYPE_REAL) * (f_max - f_min)) + f_min
        phases = torch.rand((n,), device=self.device, dtype=DTYPE_REAL) * TAU
        energy = torch.ones((n,), device=self.device, dtype=DTYPE_REAL)
        heat = torch.zeros((n,), device=self.device, dtype=DTYPE_REAL)

        self.particles["position"] = torch.cat([self.particles["position"], freqs], dim=0)
        self.particles["phase"] = torch.cat([self.particles["phase"], phases], dim=0)
        self.particles["energy"] = torch.cat([self.particles["energy"], energy], dim=0)
        self.particles["heat"] = torch.cat([self.particles["heat"], heat], dim=0)

        self._trim_particles()

    def set_targets(self, freqs: torch.Tensor, energies: Optional[torch.Tensor] = None) -> None:
        freqs = freqs.to(device=self.device, dtype=DTYPE_REAL).flatten()
        if energies is None:
            e = torch.ones_like(freqs)
        else:
            e = energies.to(device=self.device, dtype=DTYPE_REAL).flatten()
            if e.shape != freqs.shape:
                raise ValueError("energies must match freqs shape")
        self.attractors["position"] = freqs
        self.attractors["energy"] = e
        if self.attractors.get("heat", None) is None or self.attractors["heat"].numel() != freqs.numel():
            self.attractors["heat"] = torch.zeros_like(freqs)

    def _trim_particles(self) -> None:
        max_p = int(self.config.max_particles)
        n = int(self.particles["position"].shape[0])
        if n <= max_p:
            return
        e = self.particles["energy"].clamp_min(0.0)
        _, idx = torch.topk(e, k=max_p, largest=True, sorted=False)
        for k in list(self.particles.keys()):
            self.particles[k] = self.particles[k].index_select(0, idx)

    def step(self, steps: int) -> None:
        steps = int(steps)
        for _ in range(max(0, steps)):
            self.step_physics()
            # phase advance
            if self.particles["phase"].numel() > 0:
                self.particles["phase"] = (self.particles["phase"] + TAU * float(self.config.dt)) % TAU
            # energy decay (self-scaled)
            self._trim_particles()
