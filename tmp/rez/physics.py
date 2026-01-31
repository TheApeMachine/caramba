"""
Pure Physics Engine: Domain-Agnostic Thermodynamics

This is the base class that handles the raw physics loop:
- Energy, Heat, Diffusion
- Particle-Attractor interactions
- No domain-specific assumptions (no Hertz, Tokens, Embeddings)
"""

import torch
from tensordict import TensorDict
from dataclasses import dataclass
from typing import Optional

DTYPE_REAL = torch.float32


@dataclass
class PhysicsConfig:
    """Pure physics parameters. No domain assumptions."""
    dt: float = 0.01
    hold_cost: float = 0.1
    # Interaction radius: If distance > this, force is zero (Optimization)
    interaction_radius: float = 10.0
    # How much active concepts heat up their neighbors (Grammar strength)
    transition_flux: float = 1.0
    # Numerical stability
    eps: float = 1e-8


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
        # We use a "Sharpness" temperature for the softmax
        # Negative distances because softmax maximizes (we want to minimize distance)
        weights = torch.softmax(-dists, dim=1)  # [N, M]
        
        # 3. Apply Forces (Drift particles toward attractors)
        # Target position for each particle is weighted avg of attractors
        targets = self.compute_targets(weights)  # [N, ...]
        current = self.particles.get("position")  # [N, ...]
        
        drift = (targets - current)
        noise = torch.randn_like(current) * 0.1  # Simple noise for now
        
        new_pos = current + drift * self.config.dt + noise * self.config.dt
        self.particles.set("position", new_pos)
        
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
            # Initialize if needed
            self.attractors.set("energy", energy_in)
        else:
            # Exponential moving average
            new_e = current_e * 0.9 + energy_in * 0.1
            self.attractors.set("energy", new_e)
