"""
Refactored Manifold Architecture: The Great Decoupling

This addresses the "God Object" critique by splitting into:
1. ThermodynamicEngine: Pure physics (no domain-specific code)
2. SpectralManifold: Audio domain (STFT, Hertz, phase)
3. SemanticManifold: LLM domain (embeddings, tokens, grammar)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Mapping
import torch
from tensordict import TensorDict

DTYPE_REAL = torch.float32
TAU = 2.0 * math.pi


# ============================================================
# 1. Pure Physics Config (No Domain Assumptions)
# ============================================================

@dataclass
class PhysicsConfig:
    """Pure physics parameters. No Hertz, no Tokens, no Embeddings."""
    dt: float = 0.01
    hold_cost: float = 5.0
    temperature_decay: float = 0.99
    interaction_sharpness: float = 5.0  # Controls softmax sharpness in interactions
    min_gate: float = 0.01
    max_gate: float = 1.5
    eps: float = 1e-8
    max_carriers: int = 64


# ============================================================
# 2. The Pure Physics Engine (Base Class)
# ============================================================

class ThermodynamicEngine:
    """
    The Base Class. 
    Knows only about Energy, Heat, Bonds, and abstract 'Positions'.
    No domain-specific knowledge (no Hertz, no Tokens, no Embeddings).
    """
    
    def __init__(self, config: PhysicsConfig, device: torch.device):
        self.config = config
        self.device = device
        self.t = 0.0
        
        # Generic State: 'position' can be Frequency (1D) or Embedding (ND)
        self.state = TensorDict({
            "particles": TensorDict({
                "position": torch.empty(0, dtype=DTYPE_REAL, device=device), 
                "energy": torch.empty(0, dtype=DTYPE_REAL, device=device),
                "ttl": torch.empty(0, dtype=DTYPE_REAL, device=device),
            }, batch_size=[0]),
            "attractors": TensorDict({  # Renamed from 'carriers' for abstraction
                "id": torch.empty(0, dtype=torch.int64, device=device),
                "position": torch.empty(0, dtype=DTYPE_REAL, device=device),
                "gate_width": torch.empty(0, dtype=DTYPE_REAL, device=device),
                "heat": torch.empty(0, dtype=DTYPE_REAL, device=device),
                "energy": torch.empty(0, dtype=DTYPE_REAL, device=device),
            }, batch_size=[0]),
            "bonds": TensorDict({
                "strength": torch.empty(0, 0, dtype=DTYPE_REAL, device=device),
            }, batch_size=[])
        })
        
        self._next_attractor_id: int = 0

    def distance_metric(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Override this in subclasses.
        - SpectralManifold: Log-frequency with duty cycle scaling
        - SemanticManifold: Cosine distance
        """
        # Default: Euclidean distance
        return (a - b).abs()

    def step_physics(self, new_particles: Optional[TensorDict] = None):
        """
        Pure thermodynamic cycle: 
        Ingest -> Bond -> Heat -> Diffuse -> Cool
        
        This is the core physics loop. Domain-specific ingestion
        happens in subclasses before calling this.
        """
        dt = float(self.config.dt)
        
        # 1. Age out particles
        particles = self.state.get("particles")
        if particles.shape[0] > 0:
            ttl = particles.get("ttl") - dt
            alive = ttl > 0.0
            particles.set("ttl", ttl)
            # Zero energy for dead particles
            energy = particles.get("energy")
            particles.set("energy", torch.where(alive, energy, torch.zeros_like(energy)))
            self.state.set("particles", particles)
        
        # 2. Append new particles
        if new_particles is not None and new_particles.shape[0] > 0:
            old_particles = self.state.get("particles")
            if old_particles.shape[0] == 0:
                self.state.set("particles", new_particles)
            else:
                # Concatenate
                combined = TensorDict({
                    "position": torch.cat([old_particles.get("position"), new_particles.get("position")], dim=0),
                    "energy": torch.cat([old_particles.get("energy"), new_particles.get("energy")], dim=0),
                    "ttl": torch.cat([old_particles.get("ttl"), new_particles.get("ttl")], dim=0),
                }, batch_size=[old_particles.shape[0] + new_particles.shape[0]])
                self.state.set("particles", combined)
        
        attractors = self.state.get("attractors")
        if attractors.shape[0] == 0:
            self.t += dt
            return
        
        # 3. Bond particles to attractors (using distance metric)
        self._bond_particles()
        
        # 4. Update attractor energy/coherence
        self._update_attractor_energy()
        
        # 5. Temperature/excitation/gate
        self._update_temperature_excitation_gate()
        
        # 6. Costs -> heat
        self._drain_bonds_to_heat()
        self._drain_particles_to_heat()
        
        # 7. Diffuse heat
        self._diffuse_heat()
        
        # 8. Recompute temperature/excitation/gate after heat changes
        self._update_temperature_excitation_gate()
        
        self.t += dt

    def _bond_particles(self):
        """Bond particles to attractors based on distance metric."""
        particles = self.state.get("particles")
        attractors = self.state.get("attractors")
        bonds = self.state.get("bonds")
        
        n_particles = int(particles.shape[0])
        n_attractors = int(attractors.shape[0])
        
        if n_particles == 0 or n_attractors == 0:
            return
        
        # Calculate distances using domain-specific metric
        p_pos = particles.get("position")  # [N, ...]
        a_pos = attractors.get("position")  # [M, ...]
        
        # Expand for pairwise distance
        p_expanded = p_pos.unsqueeze(1)  # [N, 1, ...]
        a_expanded = a_pos.unsqueeze(0)  # [1, M, ...]
        
        # Use domain-specific distance metric
        distances = self.distance_metric(p_expanded, a_expanded)  # [N, M]
        
        # Soft assignment using softmax (differentiable)
        # Use sigmoid instead of hard gate_high for differentiability
        gate_widths = attractors.get("gate_width").unsqueeze(0)  # [1, M]
        gate_scores = torch.sigmoid((gate_widths - distances) / (gate_widths * 0.1 + self.config.eps))
        
        # Attraction strength (energy-weighted)
        p_energy = particles.get("energy").unsqueeze(1)  # [N, 1]
        a_energy = attractors.get("energy").unsqueeze(0)  # [1, M]
        
        attraction = gate_scores * p_energy * a_energy
        
        # Normalize to get bond strengths
        bond_sum = attraction.sum(dim=1, keepdim=True) + self.config.eps
        bond_strength = attraction / bond_sum  # [N, M]
        
        # Update bonds
        if bonds.get("strength").numel() == 0:
            bonds.set("strength", bond_strength)
        else:
            # Resize if needed
            old_strength = bonds.get("strength")
            if old_strength.shape[0] != n_particles or old_strength.shape[1] != n_attractors:
                bonds.set("strength", bond_strength)
            else:
                bonds.set("strength", bond_strength)
        
        self.state.set("bonds", bonds)

    def _update_attractor_energy(self):
        """Update attractor energy based on bonded particles."""
        particles = self.state.get("particles")
        attractors = self.state.get("attractors")
        bonds = self.state.get("bonds")
        
        if particles.shape[0] == 0 or attractors.shape[0] == 0:
            return
        
        bond_strength = bonds.get("strength")  # [N, M]
        p_energy = particles.get("energy").unsqueeze(1)  # [N, 1]
        
        # Attractor energy = sum of bonded particle energies
        a_energy = (bond_strength * p_energy).sum(dim=0)  # [M]
        
        attractors.set("energy", a_energy)
        self.state.set("attractors", attractors)

    def _update_temperature_excitation_gate(self):
        """Update temperature, excitation, and gate width."""
        attractors = self.state.get("attractors")
        bonds = self.state.get("bonds")
        
        if attractors.shape[0] == 0:
            return
        
        # Temperature = heat / (dof + 1)
        heat = attractors.get("heat")
        dof = bonds.get("strength").sum(dim=0) if bonds.get("strength").numel() > 0 else torch.zeros(attractors.shape[0], device=self.device)
        temperature = heat / (dof + 1.0)
        
        # Excitation relaxes toward temperature
        excitation = attractors.get("excitation")
        dt = float(self.config.dt)
        excitation = excitation + dt * (temperature - excitation)
        
        # Gate width widens with excitation
        base_width = attractors.get("gate_width")
        gate_width = base_width * (1.0 + torch.tanh(excitation))
        gate_width = gate_width.clamp(self.config.min_gate, self.config.max_gate)
        
        attractors.set("temperature", temperature)
        attractors.set("excitation", excitation)
        attractors.set("gate_width", gate_width)
        self.state.set("attractors", attractors)

    def _drain_bonds_to_heat(self):
        """Convert bond energy to heat."""
        bonds = self.state.get("bonds")
        attractors = self.state.get("attractors")
        
        if bonds.get("strength").numel() == 0 or attractors.shape[0] == 0:
            return
        
        bond_strength = bonds.get("strength")
        dt = float(self.config.dt)
        cost = float(self.config.hold_cost)
        
        # Drain proportional to bond strength
        drain = dt * cost * bond_strength.sum(dim=0)  # [M]
        
        # Convert to heat
        heat = attractors.get("heat")
        heat = heat + drain
        
        attractors.set("heat", heat)
        self.state.set("attractors", attractors)

    def _drain_particles_to_heat(self):
        """Convert particle energy to heat (via bonds)."""
        particles = self.state.get("particles")
        attractors = self.state.get("attractors")
        bonds = self.state.get("bonds")
        
        if particles.shape[0] == 0 or attractors.shape[0] == 0:
            return
        
        bond_strength = bonds.get("strength")  # [N, M]
        p_energy = particles.get("energy").unsqueeze(1)  # [N, 1]
        
        dt = float(self.config.dt)
        cost = float(self.config.hold_cost)
        
        # Drain proportional to bond strength
        drain_per_particle = dt * cost * bond_strength.sum(dim=1)  # [N]
        drain_per_particle = torch.minimum(drain_per_particle, p_energy.squeeze(1))
        
        # Update particle energy
        new_energy = p_energy.squeeze(1) - drain_per_particle
        particles.set("energy", new_energy.clamp(min=0.0))
        self.state.set("particles", particles)
        
        # Convert to heat (distributed to attractors)
        heat = attractors.get("heat")
        heat_per_attractor = (bond_strength * drain_per_particle.unsqueeze(1)).sum(dim=0)  # [M]
        heat = heat + heat_per_attractor
        
        attractors.set("heat", heat)
        self.state.set("attractors", attractors)

    def _diffuse_heat(self):
        """Diffuse heat between attractors."""
        attractors = self.state.get("attractors")
        
        if attractors.shape[0] <= 1:
            return
        
        a_pos = attractors.get("position")
        heat = attractors.get("heat")
        
        # Calculate adjacency based on position distance
        distances = self.distance_metric(a_pos.unsqueeze(1), a_pos.unsqueeze(0))
        
        # Adjacency matrix (inverse distance, normalized)
        adjacency = torch.exp(-distances / (distances.mean() + self.config.eps))
        adjacency.fill_diagonal_(0.0)
        
        # Normalize
        row_sum = adjacency.sum(dim=1) + self.config.eps
        alpha = float(self.config.dt) / float(row_sum.max().item())
        alpha = min(alpha, 0.25)  # Stability cap
        
        # Diffuse
        dheat = alpha * (adjacency @ heat - row_sum * heat)
        heat = (heat + dheat).clamp(min=0.0)
        
        attractors.set("heat", heat)
        self.state.set("attractors", attractors)


# ============================================================
# 3. The Audio Manifold (The "Fuel" - Type A)
# ============================================================

class SpectralManifold(ThermodynamicEngine):
    """
    The Audio Domain.
    Handles STFT, Hertz, Phase Wrapping.
    """
    
    def __init__(self, config: PhysicsConfig, device: torch.device):
        super().__init__(config, device)
        # Audio-specific: phase tracking
        self.state.get("particles").set("phase", torch.empty(0, dtype=DTYPE_REAL, device=device))
        self.state.get("attractors").set("phase", torch.empty(0, dtype=DTYPE_REAL, device=device))

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
        new_particles = TensorDict({
            "position": freq_bins,  # Position is Frequency (Hz)
            "energy": magnitudes * freq_bins.abs(),  # Energy = magnitude * frequency
            "phase": phases,  # Specific to Audio
            "ttl": torch.full((n,), self.config.dt * 10.0, dtype=DTYPE_REAL, device=self.device)
        }, batch_size=[n])
        
        self.step_physics(new_particles)

    def distance_metric(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Audio uses Log-Frequency metric with Duty Cycle scaling.
        This is the complex logic from the original code.
        """
        # Log-frequency distance (more perceptually accurate)
        log_a = torch.log(a.abs() + self.config.eps)
        log_b = torch.log(b.abs() + self.config.eps)
        
        # Relative distance
        rel_dist = (log_a - log_b).abs()
        
        return rel_dist


# ============================================================
# 4. The Semantic Manifold (The "Fuel" - Type B)
# ============================================================

class SemanticManifold(ThermodynamicEngine):
    """
    The LLM Domain.
    Handles Embeddings, Tokens, and Thermodynamic Grammar.
    """
    
    def __init__(self, config: PhysicsConfig, device: torch.device, embed_dim: int):
        super().__init__(config, device)
        self.embed_dim = embed_dim
        
        # --- THE CRITIQUE FIX: Bond Topology ---
        # A transition matrix [N_attractors, N_attractors]
        # If Attractor A is active, it lowers the energy cost for Attractor B
        # This implements "grammar" as energy flow
        self.transition_matrix: Optional[torch.Tensor] = None
        
        # Positional encoding for sequence position
        self.max_seq_len = 512
        self._pos_encoding_cache: Optional[torch.Tensor] = None

    def ingest_tokens(self, embeddings: torch.Tensor, positions: Optional[torch.Tensor] = None):
        """
        Ingest context tokens as particles.
        
        Args:
            embeddings: [N, embed_dim] token embeddings
            positions: [N] sequence positions (optional, auto-generated if None)
        """
        n = embeddings.shape[0]
        
        # Add Positional Encoding
        if positions is None:
            positions = torch.arange(n, dtype=DTYPE_REAL, device=self.device)
        
        pos_enc = self._get_positional_encoding(n)  # [N, embed_dim]
        
        new_particles = TensorDict({
            "position": embeddings + pos_enc,  # Position is Vector (embedding + pos)
            "energy": torch.ones(n, dtype=DTYPE_REAL, device=self.device),
            "ttl": torch.full((n,), 10.0, dtype=DTYPE_REAL, device=self.device)
        }, batch_size=[n])
        
        self.step_physics(new_particles)
        
        # Apply thermodynamic grammar after physics step
        self.apply_thermodynamic_grammar()

    def _get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """Generate positional encodings (sinusoidal)."""
        if self._pos_encoding_cache is None or self._pos_encoding_cache.shape[0] < seq_len:
            pos = torch.arange(seq_len, dtype=DTYPE_REAL, device=self.device).unsqueeze(1)
            dims = torch.arange(self.embed_dim, dtype=DTYPE_REAL, device=self.device)
            
            div_term = torch.exp(dims * -(math.log(10000.0) / self.embed_dim))
            pos_enc = torch.zeros(seq_len, self.embed_dim, dtype=DTYPE_REAL, device=self.device)
            pos_enc[:, 0::2] = torch.sin(pos * div_term[0::2])
            pos_enc[:, 1::2] = torch.cos(pos * div_term[1::2])
            
            self._pos_encoding_cache = pos_enc
        
        return self._pos_encoding_cache[:seq_len]

    def apply_thermodynamic_grammar(self):
        """
        The "Ghost Field" upgrade.
        Active attractors heat up their grammatical successors.
        
        This implements grammar as energy flow through the transition matrix.
        If "The" (high energy) is active, it lowers activation energy for "Dog".
        """
        attractors = self.state.get("attractors")
        if attractors.shape[0] == 0 or self.transition_matrix is None:
            return
        
        # 1. Get current energy of all concepts
        current_energy = attractors.get("energy")  # [N]
        
        # 2. Flow energy through the topology
        # "The" (High Energy) -> flows to -> "Dog" (Lowers Activation Energy)
        # The transition matrix defines which concepts can follow which
        flow = torch.matmul(current_energy.unsqueeze(0), self.transition_matrix).squeeze(0)  # [N]
        
        # 3. Apply bias to attractors (adds to excitation)
        # Higher flow = lower activation energy = more likely to be selected
        excitation = attractors.get("excitation")
        excitation = excitation + flow * self.config.dt * 0.1  # Scale flow
        
        attractors.set("excitation", excitation)
        self.state.set("attractors", attractors)

    def set_transition_matrix(self, matrix: torch.Tensor):
        """
        Set the grammar transition matrix.
        
        Args:
            matrix: [N_attractors, N_attractors] transition probabilities
                   matrix[i, j] = probability that attractor j follows attractor i
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Transition matrix must be square")
        
        # Ensure it matches current number of attractors
        n_attractors = self.state.get("attractors").shape[0]
        if n_attractors > 0 and matrix.shape[0] != n_attractors:
            raise ValueError(f"Transition matrix size {matrix.shape[0]} doesn't match attractors {n_attractors}")
        
        self.transition_matrix = matrix.to(device=self.device, dtype=DTYPE_REAL)

    def distance_metric(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Semantic uses Cosine Distance.
        For embeddings, cosine similarity is more appropriate than Euclidean.
        """
        # a: [N, 1, embed_dim] or [N, embed_dim]
        # b: [1, M, embed_dim] or [M, embed_dim]
        
        # Flatten if needed
        if a.dim() == 3:
            a = a.squeeze(1)  # [N, embed_dim]
        if b.dim() == 3:
            b = b.squeeze(0)  # [M, embed_dim]
        
        # Cosine similarity: 1 - cosine_sim
        # Normalize
        a_norm = a / (a.norm(dim=-1, keepdim=True) + self.config.eps)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + self.config.eps)
        
        # Pairwise cosine similarity
        if a.dim() == 2 and b.dim() == 2:
            # [N, embed_dim] @ [embed_dim, M] = [N, M]
            cosine_sim = torch.matmul(a_norm, b_norm.T)
        else:
            cosine_sim = (a_norm * b_norm).sum(dim=-1)
        
        # Distance = 1 - similarity
        distance = 1.0 - cosine_sim
        
        return distance.clamp(min=0.0)


# ============================================================
# Backward Compatibility: Alias Manifold -> SpectralManifold
# ============================================================

# For backward compatibility, alias the original name
Manifold = SpectralManifold
