from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..core.config import PhysicsConfig
from ..core.state import BatchState
from ..core.scatter import scatter_sum, segment_softmax


@dataclass
class PhysicsStepStats:
    edges: int
    heat_level: float
    sharpness: float
    energy_ratio: float


class ThermodynamicEngine:
    """Domain-agnostic thermodynamic engine with sparse interactions.

    Core design:
    - Interactions are expressed as an edge list (particle -> attractor).
    - The engine never materializes an NxM distance matrix unless a subclass chooses to.
    """

    def __init__(self, config: PhysicsConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.t = 0.0

        self.particles = BatchState.empty()
        self.attractors = BatchState.empty()

        # Homeostasis baseline (scalar)
        self._energy_baseline: Optional[torch.Tensor] = None

        self.last_stats: Optional[PhysicsStepStats] = None

    # ----------------------------
    # Hooks for subclasses
    # ----------------------------

    def candidate_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (src_idx, dst_idx) edge list for particle-attractor interactions.

        Default: all-to-all (only appropriate for small systems).
        """
        n = self.particles.n
        m = self.attractors.n
        if n == 0 or m == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        src = torch.arange(n, device=self.device, dtype=torch.long).repeat_interleave(m)
        dst = torch.arange(m, device=self.device, dtype=torch.long).repeat(n)
        return src, dst

    def distance(self, p_pos: torch.Tensor, a_pos: torch.Tensor) -> torch.Tensor:
        """Per-edge distance metric. Subclasses should override."""
        d = p_pos - a_pos
        if d.ndim == 1:
            return d.abs()
        return torch.linalg.norm(d, dim=1)

    def post_step(self) -> None:
        """Optional cleanup after physics step (e.g., TTL)."""
        return

    # ----------------------------
    # Homeostasis
    # ----------------------------

    def total_energy(self) -> torch.Tensor:
        """Total energy proxy used for homeostasis."""
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.particles.has("energy"):
            total = total + self.particles.get("energy").sum().to(torch.float32)
        if self.attractors.has("energy"):
            total = total + self.attractors.get("energy").sum().to(torch.float32)
        return total

    def _homeostasis_ratio(self) -> torch.Tensor:
        """Update baseline once and return E / baseline."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        e = self.total_energy()
        if self._energy_baseline is None:
            self._energy_baseline = e.detach().clone()
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        base = self._energy_baseline
        # Baseline timescale emerges from its own magnitude.
        alpha = dt / (dt + float(base.abs().item()) + eps)
        base_new = base * (1.0 - alpha) + e.detach() * alpha
        self._energy_baseline = base_new
        return e / (base_new + eps)

    # ----------------------------
    # Physics update
    # ----------------------------

    def step_physics(self) -> None:
        """Advance one thermodynamic step."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        n = self.particles.n
        m = self.attractors.n
        if n == 0 or m == 0:
            self.t += dt
            self.last_stats = PhysicsStepStats(edges=0, heat_level=0.0, sharpness=0.0, energy_ratio=1.0)
            return

        # Compute ratio once per step (thermostat).
        ratio = self._homeostasis_ratio().to(torch.float32)

        src, dst = self.candidate_edges()
        if src.numel() == 0:
            self.t += dt
            self.last_stats = PhysicsStepStats(edges=0, heat_level=0.0, sharpness=0.0, energy_ratio=float(ratio.item()))
            return

        p_pos = self.particles.get("position")[src]
        a_pos = self.attractors.get("position")[dst]

        dists = self.distance(p_pos, a_pos)  # [E]
        d_scale = torch.std(dists) + eps

        # Heat increases entropy: hotter => less sharp binding.
        heat_level = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.particles.has("heat") and self.particles.get("heat").numel() > 0:
            h = self.particles.get("heat")
            heat_level = (h.abs().mean() / (h.abs().mean() + 1.0 + eps)).to(torch.float32)
        elif self.attractors.has("heat") and self.attractors.get("heat").numel() > 0:
            h = self.attractors.get("heat")
            heat_level = (h.abs().mean() / (h.abs().mean() + 1.0 + eps)).to(torch.float32)

        sharpness = (1.0 / d_scale) / (1.0 + heat_level)

        logits = -dists * sharpness
        w = segment_softmax(logits, src, n, eps=eps)  # [E], sum per particle = 1

        # Targets: weighted average of attractor positions per particle.
        a_pos_full = self.attractors.get("position")
        a_gather = a_pos_full[dst]
        if a_gather.ndim == 1:
            contrib = a_gather * w
        else:
            contrib = a_gather * w.unsqueeze(1)
        targets = scatter_sum(contrib, src, n)  # [N, ...]

        cur = self.particles.get("position")
        drift = targets - cur

        # Brownian diffusion scale emerges from current dispersion and heat.
        disp = torch.std(cur) + eps
        noise = torch.randn_like(cur) * disp
        if self.particles.has("heat") and self.particles.get("heat").numel() > 0:
            h = self.particles.get("heat").abs().mean() + eps
            noise = noise * (1.0 + h / (h + 1.0))

        self.particles.set("position", cur + dt * drift + dt * noise)

        # Particle heat update: motion -> heat; heat diffuses via binding.
        if self.particles.has("heat"):
            p_heat = self.particles.get("heat")
            motion = drift.abs() if drift.ndim == 1 else torch.linalg.norm(drift, dim=1)
            motion_scale = motion.abs().mean() + eps
            p_heat = p_heat + dt * (motion / motion_scale)

            if self.attractors.has("heat"):
                a_heat = self.attractors.get("heat")
                # Project attractor heat onto particles via binding.
                h_in = scatter_sum((a_heat[dst] * w), src, n)
                h_scale = a_heat.abs().mean() + p_heat.abs().mean() + eps
                p_heat = p_heat + dt * (h_in - p_heat) / h_scale

            # Homeostatic cooling.
            h_scale = p_heat.abs().mean() + eps
            p_heat = p_heat * torch.exp(-dt * ratio.to(p_heat.dtype) / h_scale)
            self.particles.set("heat", p_heat)

        # Attractor energy and heat receive weighted inflow.
        self.update_thermodynamics(src, dst, w, ratio=ratio.to(w.dtype))

        self.post_step()
        self.t += dt

        self.last_stats = PhysicsStepStats(
            edges=int(src.numel()),
            heat_level=float(heat_level.detach().cpu().item()),
            sharpness=float(sharpness.detach().cpu().item()),
            energy_ratio=float(ratio.detach().cpu().item()),
        )

    def update_thermodynamics(self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, *, ratio: torch.Tensor) -> None:
        """Energy/heat flow between particles and attractors (edge-based)."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        m = self.attractors.n
        if m == 0:
            return

        # Energy inflow: weighted by particle energy if present; otherwise by binding mass.
        if self.particles.has("energy"):
            p_e = self.particles.get("energy")
            flow = w * p_e[src]
        else:
            flow = w

        e_in = scatter_sum(flow, dst, m)  # [M]

        a_e = self.attractors.ensure("energy", m, device=self.device, dtype=e_in.dtype)
        a_scale = a_e.abs().mean() + eps
        a_e = (a_e + dt * e_in) * torch.exp(-dt * ratio / (a_scale + eps))
        self.attractors.set("energy", a_e)

        if self.attractors.has("heat"):
            a_h = self.attractors.get("heat")
            if self.particles.has("heat"):
                p_h = self.particles.get("heat")
                h_flow = w * p_h[src]
            else:
                h_flow = w
            h_in = scatter_sum(h_flow, dst, m)
            h_scale = a_h.abs().mean() + eps
            a_h = (a_h + dt * h_in) * torch.exp(-dt * ratio / (h_scale + eps))
            self.attractors.set("heat", a_h)
