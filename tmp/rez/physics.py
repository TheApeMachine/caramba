"""
Pure Physics Engine: Domain-Agnostic Thermodynamics

This is the base class that handles the raw physics loop:
- Energy, Heat, Diffusion
- Particle-Attractor interactions
- No domain-specific assumptions (no Hertz, Tokens, Embeddings)

Note: This research explicitly rejects backpropagation. Learning emerges
from thermodynamic dynamics (energy flow, metabolic maintenance, decay).
"""

import math
import torch
from tensordict import TensorDict
from dataclasses import dataclass
from typing import Optional, Any

DTYPE_REAL = torch.float32
TAU = 2.0 * math.pi


@dataclass
class PhysicsConfig:
    """Pure physics parameters. No domain assumptions."""
    dt: float = 0.01
    # Numerical stability
    eps: float = 1e-8


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


class ThermodynamicEngine:
    """
    Domain-Agnostic Physics Engine.
    Manages Particles (Data) and Attractors (Concepts/Templates).
    
    This is the base class. Subclasses implement:
    - compute_distances(): Domain-specific distance metric
    - compute_targets(): How to calculate target positions
    """
    
    def __init__(self, config: PhysicsConfig, device: torch.device):
        self.config = config
        self.device = device
        self.t = 0.0
        
        # State containers
        self.particles = TensorDict({}, batch_size=[0])  # The "Input"
        self.attractors = TensorDict({}, batch_size=[0])  # The "Weights"
        self.bonds = TensorDict({}, batch_size=[])       # The "Interactions"

    def step_physics(self):
        """
        The Universal Loop:
        1. Calculate Distances
        2. Update Bonds (Attraction)
        3. Apply Forces (Drift)
        4. Thermodynamics (Heat/Cooling)
        """
        if self.particles.shape[0] == 0 or self.attractors.shape[0] == 0:
            self.t += self.config.dt
            return

        # 1. Distance Metric (Subclasses override this)
        dists = self.compute_distances()  # [N_particles, M_attractors]
        
        # 2. Bonding (Softmax Gravity)
        # Sharpness emerges from system scale: tighter when distances are small.
        # Negative distances because softmax maximizes (we want to minimize distance)
        dists_scale = torch.std(dists) + self.config.eps
        # Gravity strengthens as heat rises (concentration counters entropy)
        heat_level = 0.0
        if "heat" in self.attractors.keys() and self.attractors.get("heat").numel() > 0:
            h = self.attractors.get("heat")
            h_scale = torch.mean(h.abs()) + self.config.eps
            heat_level = float((h_scale / (h_scale + 1.0)).item())
        elif "heat" in self.particles.keys() and self.particles.get("heat").numel() > 0:
            h = self.particles.get("heat")
            h_scale = torch.mean(h.abs()) + self.config.eps
            heat_level = float((h_scale / (h_scale + 1.0)).item())
        sharpness = (1.0 / dists_scale) * (1.0 + heat_level)
        weights = torch.softmax(-dists * sharpness, dim=1)  # [N, M]
        
        # 3. Apply Forces (Drift particles toward attractors)
        # Target position for each particle is weighted avg of attractors
        targets = self.compute_targets(weights)  # [N, ...]
        current = self.particles.get("position")  # [N, ...]
        
        drift = (targets - current)
        # Noise emerges from system dispersion (no fixed scale)
        noise_scale = torch.std(current) + self.config.eps
        noise = torch.randn_like(current) * noise_scale
        
        new_pos = current + drift * self.config.dt + noise * self.config.dt
        self.particles.set("position", new_pos)
        
        # Heat transport: kinetic activity raises particle heat, which diffuses via weights
        if "heat" in self.particles.keys():
            p_heat = self.particles.get("heat")
            drift_mag = drift.abs() if drift.dim() == 1 else torch.norm(drift, dim=1)
            d_scale = torch.mean(drift_mag.abs()) + self.config.eps
            p_heat = p_heat + (drift_mag / d_scale) * self.config.dt
            if "heat" in self.attractors.keys():
                a_heat = self.attractors.get("heat")
                h_scale = torch.mean(a_heat.abs()) + self.config.eps
                p_heat = p_heat + self.config.dt * (torch.matmul(weights, a_heat) - p_heat) / h_scale
            # Cooling proportional to current heat scale
            h_scale = torch.mean(p_heat.abs()) + self.config.eps
            p_heat = p_heat * torch.exp(-self.config.dt / h_scale)
            self.particles.set("heat", p_heat)
            # Heat raises kinetic activity (noise scale)
            noise_scale = noise_scale * (1.0 + torch.mean(p_heat.abs()))

        # 4. Update Energy/Heat (Abstracted)
        self.update_thermodynamics(weights)
        
        self.t += self.config.dt

    def compute_distances(self) -> torch.Tensor:
        """
        Subclasses must define distance metric.
        Returns: [N_particles, M_attractors] distance matrix
        """
        raise NotImplementedError("Subclasses must define distance metric")

    def compute_targets(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Subclasses must define how to compute target positions.
        Args:
            weights: [N_particles, M_attractors] softmax weights
        Returns:
            [N_particles, ...] target positions
        """
        raise NotImplementedError("Subclasses must define target calculation")

    def update_thermodynamics(self, weights: torch.Tensor):
        """
        Basic energy transfer: Particles give energy to Attractors.
        Energy = sum(weights) per attractor.
        """
        # Energy in = sum of particle weights per attractor
        energy_in = weights.sum(dim=0)  # [M_attractors]
        
        current_e = self.attractors.get("energy")
        if current_e.numel() == 0:
            self.attractors.set("energy", energy_in)
        else:
            # EMA where alpha emerges from system energy scale
            dt = float(self.config.dt)
            energy_scale = torch.mean(current_e.abs()) + self.config.eps
            tau = 1.0 / energy_scale
            alpha = dt / (tau + dt)
            new_e = current_e * (1.0 - alpha) + energy_in * alpha
            self.attractors.set("energy", new_e)

        # Heat transport follows energy flow when heat is present
        if "heat" in self.attractors.keys():
            heat_in = energy_in
            if "heat" in self.particles.keys():
                p_heat = self.particles.get("heat")
                heat_in = torch.matmul(weights.T, p_heat)
            current_h = self.attractors.get("heat")
            dt = float(self.config.dt)
            h_scale = torch.mean(current_h.abs()) + self.config.eps
            tau = 1.0 / h_scale
            alpha = dt / (tau + dt)
            new_h = current_h * (1.0 - alpha) + heat_in * alpha
            self.attractors.set("heat", new_h)


# ============================================================
# Audio Domain: SpectralManifold
# ============================================================

class SpectralManifold(ThermodynamicEngine):
    """
    The Audio Domain.
    Handles STFT, Hertz, Phase Wrapping.
    """

    def __init__(self, config: PhysicsConfig, device: torch.device):
        super().__init__(config, device)
        # Audio-specific: phase tracking
        self.particles.set("phase", torch.empty(0, dtype=DTYPE_REAL, device=device))
        self.attractors.set("phase", torch.empty(0, dtype=DTYPE_REAL, device=device))
        self.particles.set("heat", torch.empty(0, dtype=DTYPE_REAL, device=device))
        self.attractors.set("heat", torch.empty(0, dtype=DTYPE_REAL, device=device))

    def ingest_frame(self, freq_bins: torch.Tensor, magnitudes: torch.Tensor, phases: torch.Tensor):
        """
        Type-safe ingestion for Audio.
        Converts STFT data into 'Particles'.

        Args:
            freq_bins: [N] frequencies in Hz
            magnitudes: [N] magnitudes
            phases: [N] phases in radians
        """
        n = freq_bins.shape[0]
        new_particles = TensorDict(
            {
                "position": freq_bins,  # Position is Frequency (Hz)
                "energy": magnitudes * freq_bins.abs(),  # Energy = magnitude * frequency
                "phase": phases,  # Specific to Audio
                "ttl": torch.full((n,), self.config.dt * 10.0, dtype=DTYPE_REAL, device=self.device),
            },
            batch_size=[n],
        )

        self.step_physics()
        # Append new particles after step to keep physics core minimal
        if self.particles.shape[0] == 0:
            self.particles = new_particles
        else:
            self.particles = TensorDict(
                {
                    "position": torch.cat([self.particles.get("position"), new_particles.get("position")], dim=0),
                    "energy": torch.cat([self.particles.get("energy"), new_particles.get("energy")], dim=0),
                    "phase": torch.cat([self.particles.get("phase"), new_particles.get("phase")], dim=0),
                    "ttl": torch.cat([self.particles.get("ttl"), new_particles.get("ttl")], dim=0),
                },
                batch_size=[self.particles.shape[0] + n],
            )

    def compute_distances(self) -> torch.Tensor:
        """
        Audio uses Log-Frequency metric with Duty Cycle scaling.
        """
        p = self.particles.get("position")  # [N]
        a = self.attractors.get("position")  # [M]
        if p.numel() == 0 or a.numel() == 0:
            return torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device)
        log_p = torch.log(p.abs() + self.config.eps).unsqueeze(1)
        log_a = torch.log(a.abs() + self.config.eps).unsqueeze(0)
        return (log_p - log_a).abs()

    def compute_targets(self, weights: torch.Tensor) -> torch.Tensor:
        a_pos = self.attractors.get("position")
        return torch.mm(weights, a_pos.unsqueeze(1)).squeeze(1) if a_pos.dim() == 1 else torch.mm(weights, a_pos)

    def step_physics(self):
        """
        Run physics step and clean up expired particles.
        """
        super().step_physics()
        if self.particles.shape[0] == 0 or "ttl" not in self.particles.keys():
            return
        ttl = self.particles.get("ttl") - self.config.dt
        alive = ttl > 0
        if alive.all():
            self.particles.set("ttl", ttl)
            return
        if alive.any():
            self.particles = TensorDict(
                {key: self.particles.get(key)[alive] for key in self.particles.keys()},
                batch_size=[int(alive.sum().item())],
            )
            self.particles.set("ttl", ttl[alive])
        else:
            self.particles = TensorDict({}, batch_size=[0])

    def output_state(self) -> OutputState:
        """Return unified output state for spectral generation."""
        particles = self.particles.get("position") if self.particles.shape[0] > 0 else None
        targets = self.attractors.get("position") if self.attractors.shape[0] > 0 else None
        return OutputState(audio_particles=particles, audio_targets=targets)
