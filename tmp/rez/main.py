"""
Resonant Compression Systems — Core Engine
===========================================

Implementation follows the paper specification exactly.
GPU-ready via PyTorch tensors.

Key Physics:
- Oscillators: phase φ, frequency ω, amplitude A, phasor z = A·e^(iφ)
- Carriers: complex amplitude c = r·e^(iψ), with gated capture
- Presence matrix P: elastic bonds that must be sustained by resonance
- Coupling strength emerges from phase alignment (antenna principle)

The system starts empty. Carriers nucleate from unbound oscillator energy.
Mitosis occurs when interference destabilizes a carrier.
Dissolution occurs when a carrier fails to capture sufficient energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Callable
from pathlib import Path

import torch
import torch.nn.functional as F

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_REAL = torch.float32
DTYPE_COMPLEX = torch.complex64


# =============================================================================
# Physics Constants (from paper, Section 4)
# These are properties of the medium, not tunable hyperparameters
# =============================================================================

@dataclass
class PhysicsConfig:
    """
    Physical constants of the resonant system.
    
    Most values are DERIVED from the physics, not arbitrary tuning parameters.
    The coupling strength is NOT a constant - it emerges from wave alignment.
    """
    
    # =========================================================================
    # SIMULATION (DERIVED FROM SIGNAL FREQUENCIES)
    # =========================================================================
    # dt = 0.005s gives 200 samples/s
    # For signals up to 4.5 Hz, need >9 samples/cycle (Nyquist)
    # 200 / 4.5 = 44 samples/cycle - good margin for numerical stability
    dt: float = 0.005
    
    # =========================================================================
    # CARRIER GATE (DERIVED FROM GATE GEOMETRY)
    # =========================================================================
    # The gate is open when cos(ψ) >= 0, meaning it's open for π radians
    # (half the cycle, from -π/2 to +π/2).
    # Gate width W = π radians
    gate_width: float = math.pi
    
    # =========================================================================
    # CARRIER PHYSICS (Eq. 10-11)
    # =========================================================================
    # gamma: Damping coefficient. Carrier decays as e^(-γt) without input.
    # This determines how long a carrier "remembers" without new input.
    gamma: float = 2.0
    
    # beta: Saturation. Limits max amplitude via cubic term β|c|²c.
    # This prevents unbounded energy accumulation.
    beta: float = 0.5
    
    # =========================================================================
    # ELASTIC BONDS (Eq. 19)
    # =========================================================================
    # These control bond dynamics - how fast bonds form/decay
    tau_p: float = 6.0      # Bond timescale
    alpha_p: float = 2.0    # Reinforcement rate
    lambda_p: float = 0.25  # Decay rate
    
    # p_snap: Bond snap threshold. Below this, bond breaks.
    # 1% of max bond strength is a reasonable "noise floor".
    p_snap: float = 0.01
    
    # =========================================================================
    # COHERENCE AND MITOSIS (Eq. 24-25)
    # =========================================================================
    d_threshold: float = 0.95  # Division threshold
    mitosis_window: int = 50   # Persistence window (steps)
    
    # =========================================================================
    # METABOLISM
    # =========================================================================
    intake_tau: float = 2.5    # EMA time constant for intake
    metabolic_cost: float = 0.2  # Minimum intake to survive
    starve_window: int = 300   # Steps below threshold before dissolution
    
    # =========================================================================
    # GENESIS
    # =========================================================================
    # unbound_threshold: Oscillator is "unbound" if max bond < this.
    unbound_threshold: float = 0.1
    
    # genesis_pressure: Summed unbound amplitude needed to nucleate.
    genesis_pressure: float = 1.25
    
    # genesis_coherence_threshold: Unbound oscillators must show phase alignment
    # to nucleate a carrier. This prevents genesis from incoherent noise.
    genesis_coherence_threshold: float = 0.5
    
    # =========================================================================
    # GATE WIDTH EVOLUTION (Specialization)
    # =========================================================================
    # Gate width evolves based on coherence history:
    # - High coherence → narrower gate (more selective)
    # - Low coherence → wider gate (more exploratory)
    # This creates a spectrum from broad exploratory carriers to narrow specialists.
    
    # Minimum gate width (maximum specialization)
    gate_width_min: float = math.pi / 4  # Quarter cycle
    
    # Maximum gate width (maximum exploration)
    gate_width_max: float = math.pi * 1.5  # 3/4 cycle
    
    # Rate of gate narrowing under sustained coherence
    gate_narrow_rate: float = 0.1
    
    # Rate of gate widening under low coherence
    gate_widen_rate: float = 0.05
    
    # =========================================================================
    # NOISE FLOOR
    # =========================================================================
    # Below this, things are considered "dead" / non-existent.
    noise_floor: float = 1e-3


# =============================================================================
# Input Signal Protocol
# =============================================================================

@dataclass
class Signal:
    """
    An input signal to the system.
    
    Signals are the external drive - they come from the environment.
    They have frequency, phase, and amplitude.
    """
    freq_hz: float
    phase: float  # radians, [0, 2π)
    amplitude: float
    duration_s: float  # How long this signal is driven
    
    @property
    def omega(self) -> float:
        """Angular frequency in rad/s."""
        return 2 * math.pi * self.freq_hz
    
    @property
    def phasor(self) -> complex:
        """Complex phasor representation."""
        return self.amplitude * (math.cos(self.phase) + 1j * math.sin(self.phase))


class SignalSource(Protocol):
    """Protocol for signal sources (can be stochastic stream, dataset, etc.)."""
    
    def get_signals(self, t: float, dt: float) -> list[Signal]:
        """Return signals active at time t."""
        ...


# =============================================================================
# Core Tensors
# =============================================================================

class OscillatorState:
    """
    State of all oscillators (input signals internalized).
    
    Stored as tensors for GPU computation.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        # Each oscillator has: phase, omega (angular freq), amplitude, drive_remaining
        self.phases: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self.omegas: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self.amplitudes: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self.drive_remaining: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self._next_id = 0
        self._ids: list[int] = []  # Track IDs for sparse operations
    
    @property
    def n(self) -> int:
        """Number of oscillators."""
        return len(self._ids)
    
    @property
    def phasors(self) -> torch.Tensor:
        """Complex phasors z = A·e^(iφ)."""
        return self.amplitudes * torch.exp(1j * self.phases.to(DTYPE_COMPLEX))
    
    def add(self, signal: Signal) -> int:
        """Add an oscillator from an input signal. Returns its ID."""
        oid = self._next_id
        self._next_id += 1
        self._ids.append(oid)
        
        # Expand tensors
        self.phases = torch.cat([
            self.phases, 
            torch.tensor([signal.phase], dtype=DTYPE_REAL, device=self.device)
        ])
        self.omegas = torch.cat([
            self.omegas,
            torch.tensor([signal.omega], dtype=DTYPE_REAL, device=self.device)
        ])
        self.amplitudes = torch.cat([
            self.amplitudes,
            torch.tensor([0.0], dtype=DTYPE_REAL, device=self.device)  # Starts at 0, ramps up
        ])
        self.drive_remaining = torch.cat([
            self.drive_remaining,
            torch.tensor([signal.duration_s], dtype=DTYPE_REAL, device=self.device)
        ])
        
        return oid
    
    def remove(self, indices: torch.Tensor) -> None:
        """Remove oscillators at given indices."""
        if indices.numel() == 0:
            return
        
        keep_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        
        self.phases = self.phases[keep_mask]
        self.omegas = self.omegas[keep_mask]
        self.amplitudes = self.amplitudes[keep_mask]
        self.drive_remaining = self.drive_remaining[keep_mask]
        self._ids = [self._ids[i] for i in range(len(self._ids)) if keep_mask[i]]
    
    def step_amplitudes(self, dt: float, damping: float = 1.5, saturation: float = 0.2) -> None:
        """
        Update amplitudes based on drive status.
        
        Driven oscillators ramp up, undriven oscillators decay.
        From paper Eq. 17: dA/dt = -α(A - A₀) + I(t) - ρA³
        Simplified: driven grows, undriven decays with saturation.
        """
        # Decrease drive remaining
        self.drive_remaining = torch.clamp(self.drive_remaining - dt, min=0.0)
        
        # Drive is 1 when active, 0 when expired
        drive = (self.drive_remaining > 0).float()
        
        # Amplitude dynamics: driven growth - damping - saturation
        dA = drive - damping * self.amplitudes - saturation * (self.amplitudes ** 3)
        self.amplitudes = torch.clamp(self.amplitudes + dA * dt, min=0.0)


class CarrierState:
    """
    State of all carriers.
    
    Carriers are spectral hypotheses - pulse-gated antennas that capture energy 
    from aligned oscillators. Each carrier has:
    - Complex amplitude c = r·e^(iψ)
    - Intrinsic frequency ω (center of spectral hypothesis)
    - Gate width W (perceptual tolerance - narrows with specialization)
    - Coherence history D (for tracking conflict)
    """
    
    def __init__(self, device: torch.device = DEVICE, initial_gate_width: float = math.pi):
        self.device = device
        self.initial_gate_width = initial_gate_width
        
        # Complex amplitudes c = r·e^(iψ)
        self.c: torch.Tensor = torch.empty(0, dtype=DTYPE_COMPLEX, device=device)
        # Intrinsic frequencies (for rotation) - this is the spectral center
        self.omegas: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Per-carrier gate width - starts at initial_gate_width, evolves with coherence
        self.gate_widths: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Coherence EMA - tracks quality of captured signal
        self.coherence_ema: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Intake EMA for metabolism (now coherence-weighted)
        self.intake_ema: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Starve counter
        self.starve_steps: torch.Tensor = torch.empty(0, dtype=torch.int32, device=device)
        # Coherence history for mitosis detection
        self.d_below_threshold: torch.Tensor = torch.empty(0, dtype=torch.int32, device=device)
        # Birth time for lifetime tracking
        self.birth_t: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        
        self._next_id = 0
        self._ids: list[int] = []
        self._names: list[str] = []
    
    @property
    def m(self) -> int:
        """Number of carriers."""
        return len(self._ids)
    
    @property
    def phases(self) -> torch.Tensor:
        """Carrier phases ψ = angle(c)."""
        return torch.angle(self.c)
    
    @property
    def energies(self) -> torch.Tensor:
        """Carrier energies |c|."""
        return torch.abs(self.c)
    
    def gate(self) -> torch.Tensor:
        """
        Gate function G(ψ) from Eq. 9.
        
        Open when cos(ψ) >= 0 (half-cycle, 50% duty).
        This is the pulse antenna - determines WHEN capture happens.
        """
        return (torch.cos(self.phases) >= 0).float()
    
    def add(self, phase: float, omega_hz: float, t: float, gate_width: Optional[float] = None) -> int:
        """
        Add a new carrier. Returns its ID.
        
        Args:
            phase: Initial phase
            omega_hz: Intrinsic frequency in Hz (spectral center of hypothesis)
            t: Birth time
            gate_width: Initial gate width (perceptual tolerance). If None, uses default.
        """
        cid = self._next_id
        self._next_id += 1
        self._ids.append(cid)
        self._names.append(f"C{cid}")
        
        omega = 2 * math.pi * omega_hz
        gw = gate_width if gate_width is not None else self.initial_gate_width
        
        # Initial complex amplitude: small magnitude, given phase
        c_init = 0.1 * (math.cos(phase) + 1j * math.sin(phase))
        
        self.c = torch.cat([
            self.c,
            torch.tensor([c_init], dtype=DTYPE_COMPLEX, device=self.device)
        ])
        self.omegas = torch.cat([
            self.omegas,
            torch.tensor([omega], dtype=DTYPE_REAL, device=self.device)
        ])
        self.gate_widths = torch.cat([
            self.gate_widths,
            torch.tensor([gw], dtype=DTYPE_REAL, device=self.device)
        ])
        self.coherence_ema = torch.cat([
            self.coherence_ema,
            torch.tensor([1.0], dtype=DTYPE_REAL, device=self.device)  # Start optimistic
        ])
        self.intake_ema = torch.cat([
            self.intake_ema,
            torch.tensor([0.5], dtype=DTYPE_REAL, device=self.device)  # Start with some credit
        ])
        self.starve_steps = torch.cat([
            self.starve_steps,
            torch.tensor([0], dtype=torch.int32, device=self.device)
        ])
        self.d_below_threshold = torch.cat([
            self.d_below_threshold,
            torch.tensor([0], dtype=torch.int32, device=self.device)
        ])
        self.birth_t = torch.cat([
            self.birth_t,
            torch.tensor([t], dtype=DTYPE_REAL, device=self.device)
        ])
        
        return cid
    
    def remove(self, indices: torch.Tensor) -> list[dict]:
        """Remove carriers at given indices. Returns death event info."""
        if indices.numel() == 0:
            return []
        
        events = []
        for idx in indices.tolist():
            events.append({
                "id": self._ids[idx],
                "name": self._names[idx],
                "birth_t": float(self.birth_t[idx]),
                "final_energy": float(self.energies[idx]),
                "intake": float(self.intake_ema[idx]),
                "final_gate_width": float(self.gate_widths[idx]),
                "final_coherence": float(self.coherence_ema[idx]),
            })
        
        keep_mask = torch.ones(self.m, dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        
        self.c = self.c[keep_mask]
        self.omegas = self.omegas[keep_mask]
        self.gate_widths = self.gate_widths[keep_mask]
        self.coherence_ema = self.coherence_ema[keep_mask]
        self.intake_ema = self.intake_ema[keep_mask]
        self.starve_steps = self.starve_steps[keep_mask]
        self.d_below_threshold = self.d_below_threshold[keep_mask]
        self.birth_t = self.birth_t[keep_mask]
        self._ids = [self._ids[i] for i in range(len(self._ids)) if keep_mask[i]]
        self._names = [self._names[i] for i in range(len(self._names)) if keep_mask[i]]
        
        return events


class PresenceMatrix:
    """
    Presence matrix P ∈ ℝ≥0^(N×M).
    
    P[i,k] is the elastic bond strength between oscillator i and carrier k.
    Bonds must be continuously sustained by resonant capture.
    
    This is stored as a dense matrix for simplicity, but should be sparse
    for large N, M.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.P: torch.Tensor = torch.empty(0, 0, dtype=DTYPE_REAL, device=device)
    
    def resize(self, n_osc: int, n_car: int) -> None:
        """Resize the matrix, preserving existing values."""
        if n_osc == 0 or n_car == 0:
            self.P = torch.empty(n_osc, n_car, dtype=DTYPE_REAL, device=self.device)
            return
        
        old_n, old_m = self.P.shape if self.P.numel() > 0 else (0, 0)
        
        if old_n == n_osc and old_m == n_car:
            return
        
        new_P = torch.zeros(n_osc, n_car, dtype=DTYPE_REAL, device=self.device)
        
        if old_n > 0 and old_m > 0:
            copy_n = min(old_n, n_osc)
            copy_m = min(old_m, n_car)
            new_P[:copy_n, :copy_m] = self.P[:copy_n, :copy_m]
        
        self.P = new_P
    
    def remove_oscillators(self, indices: torch.Tensor) -> None:
        """Remove rows for deleted oscillators."""
        if indices.numel() == 0 or self.P.numel() == 0:
            return
        keep_mask = torch.ones(self.P.shape[0], dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        self.P = self.P[keep_mask]
    
    def remove_carriers(self, indices: torch.Tensor) -> None:
        """Remove columns for deleted carriers."""
        if indices.numel() == 0 or self.P.numel() == 0:
            return
        keep_mask = torch.ones(self.P.shape[1], dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        self.P = self.P[:, keep_mask]
    
    def add_oscillator(self) -> None:
        """Add a row for a new oscillator (zeros)."""
        if self.P.numel() == 0:
            return
        new_row = torch.zeros(1, self.P.shape[1], dtype=DTYPE_REAL, device=self.device)
        self.P = torch.cat([self.P, new_row], dim=0)
    
    def add_carrier(self) -> None:
        """Add a column for a new carrier (zeros)."""
        if self.P.numel() == 0:
            return
        new_col = torch.zeros(self.P.shape[0], 1, dtype=DTYPE_REAL, device=self.device)
        self.P = torch.cat([self.P, new_col], dim=1)
    
    def seed_bond(self, osc_idx: int, car_idx: int, strength: float) -> None:
        """Seed a bond between oscillator and carrier."""
        if self.P.numel() == 0:
            return
        self.P[osc_idx, car_idx] = strength
    
    def nnz(self) -> int:
        """Number of non-zero entries."""
        if self.P.numel() == 0:
            return 0
        return int((self.P > 0).sum().item())


# =============================================================================
# Coupling Computation - THE ANTENNA PRINCIPLE
# =============================================================================

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """
    Wrap angle to [-π, π] range.
    
    This is critical for computing the shortest angular distance.
    Without this, the distance between 0.1π and 1.9π would be computed
    as 1.8π instead of the correct 0.2π.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_tuning_strength(
    carrier_phases: torch.Tensor,  # [M] - carrier gate center positions
    osc_phases: torch.Tensor,      # [N] - oscillator phases
    gate_width: float              # Width of the gate window in radians
) -> torch.Tensor:
    """
    Compute coupling strength using the RADIO TUNING principle (uniform gate width).
    
    This is NOT a simple cosine - it's a Gaussian falloff that creates
    the "lock-on" feel of tuning a radio dial:
    
    - T ≈ 1: Perfectly aligned - strong coupling, clear signal
    - T ≈ 0.5: Slightly off - partial coupling, signal + noise  
    - T ≈ 0: Far off - no coupling, just noise
    
    The coupling strength T is computed as:
        T = exp(-(diff² / σ))
    
    Where:
    - diff = shortest angular distance between gate center and oscillator peak
    - σ = tuning sharpness, derived from gate width: σ = (W/2)²
    
    This ensures that when the oscillator peak moves outside the physical
    "open" window of the gate, coupling drops significantly.
    
    Returns: [N, M] tensor of tuning strengths in [0, 1]
    """
    # carrier_phases: [M] - these are the gate CENTER positions (ψ)
    # The gate is open when cos(ψ) >= 0, so center is at ψ = 0 relative to carrier phase
    # In absolute terms, gate center = carrier_phase
    gate_centers = carrier_phases  # [M]
    
    # osc_phases: [N] - oscillator phases
    # For phasor z = A·e^(iφ), real part is A·cos(φ), which peaks at φ = 0
    # So oscillator peak position = osc_phase
    osc_peaks = osc_phases  # [N]
    
    # Compute phase difference with proper wrapping
    # Broadcasting: [1, M] - [N, 1] = [N, M]
    raw_diff = gate_centers.unsqueeze(0) - osc_peaks.unsqueeze(1)
    diff = wrap_to_pi(raw_diff)  # Shortest angular distance
    
    # Tuning sharpness σ derived from gate width
    # If gate is open for W radians, we want T to drop significantly
    # when the oscillator peak is outside that window.
    # Setting σ = (W/2)² means at diff = W/2, T = exp(-1) ≈ 0.37
    sigma = (gate_width / 2) ** 2
    
    # Gaussian tuning strength
    T = torch.exp(-(diff ** 2) / sigma)
    
    return T


def compute_tuning_strength_per_carrier(
    carrier_phases: torch.Tensor,    # [M] - carrier gate center positions
    osc_phases: torch.Tensor,        # [N] - oscillator phases
    carrier_gate_widths: torch.Tensor  # [M] - per-carrier gate widths (specialization)
) -> torch.Tensor:
    """
    Compute coupling strength with PER-CARRIER gate widths.
    
    This is the key to EMERGENT SPECIALIZATION:
    - Narrow-gate carriers are specialists: sharp tuning, strong lock-on
    - Wide-gate carriers are generalists: loose tuning, broader capture
    
    Each carrier's σ is derived from its individual gate width:
        σ_k = (W_k / 2)²
    
    Returns: [N, M] tensor of tuning strengths in [0, 1]
    """
    if carrier_phases.numel() == 0 or osc_phases.numel() == 0:
        device = carrier_phases.device if carrier_phases.numel() > 0 else osc_phases.device
        return torch.empty(osc_phases.numel(), carrier_phases.numel(), 
                          dtype=DTYPE_REAL, device=device)
    
    # Compute phase difference with proper wrapping
    # Broadcasting: [1, M] - [N, 1] = [N, M]
    raw_diff = carrier_phases.unsqueeze(0) - osc_phases.unsqueeze(1)
    diff = wrap_to_pi(raw_diff)  # [N, M]
    
    # Per-carrier tuning sharpness: σ_k = (W_k / 2)²
    # carrier_gate_widths: [M] -> sigma: [M]
    sigma = (carrier_gate_widths / 2) ** 2  # [M]
    
    # Broadcasting: diff² [N, M] / sigma [1, M] = [N, M]
    T = torch.exp(-(diff ** 2) / sigma.unsqueeze(0))
    
    return T


def compute_alignment(carrier_phases: torch.Tensor, osc_phases: torch.Tensor) -> torch.Tensor:
    """
    DEPRECATED - kept for backward compatibility with tests.
    Use compute_tuning_strength instead.
    
    This cosine-based alignment is "too smooth" - it doesn't create
    the sharp "lock-on" feel of real radio tuning.
    """
    phase_diff = carrier_phases.unsqueeze(0) - osc_phases.unsqueeze(1)
    return torch.cos(phase_diff)


def compute_carrier_drive(
    osc_phasors: torch.Tensor,   # [N] complex
    P: torch.Tensor,             # [N, M]
    carrier_gates: torch.Tensor, # [M] binary gate (0 or 1)
    tuning: torch.Tensor         # [N, M] tuning strength from compute_tuning_strength
) -> torch.Tensor:
    """
    Compute gated drive to each carrier with tuning-based coupling.
    
    u_k = G(ψ_k) · Σᵢ T_ik · P_ik · z_i
    
    Where:
    - G(ψ_k) = gate function (0 or 1) - determines WHEN capture happens
    - T_ik = tuning strength - determines HOW STRONGLY oscillator couples (antenna principle)
    - P_ik = bond strength - determines the established connection weight
    - z_i = oscillator phasor
    
    The key insight: coupling strength T is NOT a constant - it emerges from
    the alignment between the carrier's gate and the oscillator's phase.
    
    Returns: [M] complex tensor of carrier drives
    """
    if P.numel() == 0 or osc_phasors.numel() == 0:
        return torch.empty(0, dtype=DTYPE_COMPLEX, device=P.device)
    
    # Effective coupling: tuning strength × bond strength
    # T: [N, M], P: [N, M] -> effective: [N, M]
    effective_coupling = tuning * P
    
    # u = (T⊙P)^T @ z  (element-wise product then matrix-vector)
    # effective: [N, M], z: [N] -> u: [M]
    u_raw = torch.matmul(effective_coupling.T.to(DTYPE_COMPLEX), osc_phasors)
    
    # Apply gate - only capture when gate is open
    u_gated = carrier_gates.to(DTYPE_COMPLEX) * u_raw
    
    return u_gated


def compute_back_influence(
    carrier_c: torch.Tensor,  # [M] complex
    P: torch.Tensor,          # [N, M]
    tuning: torch.Tensor      # [N, M] tuning strength
) -> torch.Tensor:
    """
    Compute back-influence from carriers to oscillators.
    
    g_i = Σ_k T_ik · P_ik · c_k
    
    This is how oscillators "feel" the carriers they're connected to.
    The influence is modulated by the tuning strength - oscillators
    only feel carriers they're well-aligned with.
    
    Returns: [N] complex tensor of back-influences
    """
    if P.numel() == 0 or carrier_c.numel() == 0:
        return torch.empty(0, dtype=DTYPE_COMPLEX, device=P.device)
    
    # Effective coupling: tuning × bond
    effective_coupling = tuning * P
    
    # g = (T⊙P) @ c
    # effective: [N, M], c: [M] -> g: [N]
    g = torch.matmul(effective_coupling.to(DTYPE_COMPLEX), carrier_c)
    
    return g


def compute_phase_influence(
    g: torch.Tensor,           # [N] complex back-influence (already tuning-weighted)
    osc_phases: torch.Tensor,  # [N]
) -> torch.Tensor:
    """
    Compute phase velocity modification from coupling.
    
    Δφ̇_i = Im(g_i · e^(-iφ_i))
    
    NOTE: There is no κ (kappa) constant here. The coupling strength
    is already embedded in g through the tuning strength T.
    The back-influence g was computed with tuning-weighted bonds,
    so the phase pull is automatically stronger for well-aligned pairs.
    
    This creates Kuramoto-like synchronization dynamics where the
    coupling strength emerges from the physics, not a constant.
    
    Returns: [N] tensor of phase velocity modifications
    """
    if g.numel() == 0:
        return torch.empty(0, dtype=DTYPE_REAL, device=g.device)
    
    # g_i · e^(-iφ_i)
    rotated = g * torch.exp(-1j * osc_phases.to(DTYPE_COMPLEX))
    
    # Take imaginary part - this is the phase pull
    phase_influence = rotated.imag
    
    return phase_influence.to(DTYPE_REAL)


# =============================================================================
# Spectral Profile Computation
# =============================================================================

def compute_spectral_profiles(
    osc_omegas: torch.Tensor,    # [N] oscillator frequencies
    osc_amplitudes: torch.Tensor, # [N] oscillator amplitudes
    P: torch.Tensor,              # [N, M] presence matrix
    tuning: torch.Tensor          # [N, M] tuning strength
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute spectral profile for each carrier.
    
    A carrier's spectral profile describes which frequencies it represents.
    This is an OBSERVABLE property, not a learned parameter.
    
    Effective weights: w_ik = P_ik · A_i · T_ik
    
    Returns:
        omega_center: [M] spectral center of mass for each carrier
        spectral_variance: [M] spread of frequencies
        is_multimodal: [M] bool tensor indicating bimodal/multimodal spectra
    """
    if P.numel() == 0 or osc_omegas.numel() == 0:
        device = P.device if P.numel() > 0 else osc_omegas.device
        empty_real = torch.empty(0, dtype=DTYPE_REAL, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        return empty_real, empty_real, empty_bool
    
    # Effective weights: w_ik = P_ik * A_i * T_ik
    # P: [N, M], A: [N], T: [N, M] -> W: [N, M]
    W = P * osc_amplitudes.unsqueeze(1) * tuning
    
    # Normalize weights per carrier
    W_sum = W.sum(dim=0, keepdim=True) + 1e-8  # [1, M]
    W_norm = W / W_sum  # [N, M]
    
    # Spectral center of mass: ω̄_k = Σᵢ w_ik · ωᵢ / Σᵢ w_ik
    omega_center = (W_norm * osc_omegas.unsqueeze(1)).sum(dim=0)  # [M]
    
    # Spectral variance: Var_k = Σᵢ w_ik · (ωᵢ - ω̄_k)² / Σᵢ w_ik
    omega_diff = osc_omegas.unsqueeze(1) - omega_center.unsqueeze(0)  # [N, M]
    spectral_variance = (W_norm * (omega_diff ** 2)).sum(dim=0)  # [M]
    
    # Detect multimodality: check if variance is "too high" relative to mean
    # A carrier with two distinct frequency clusters will have high variance
    # relative to what a single Gaussian would predict.
    # Heuristic: variance > (range/4)² suggests multimodality
    omega_range = osc_omegas.max() - osc_omegas.min() + 1e-8
    threshold = (omega_range / 4) ** 2
    is_multimodal = spectral_variance > threshold
    
    return omega_center, spectral_variance, is_multimodal


def partition_oscillators_by_frequency(
    osc_omegas: torch.Tensor,    # [N] oscillator frequencies
    P_col: torch.Tensor,          # [N] bond strengths for ONE carrier
    tuning_col: torch.Tensor,     # [N] tuning strengths for ONE carrier
    osc_amplitudes: torch.Tensor  # [N] oscillator amplitudes
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Partition bonded oscillators into two frequency clusters.
    
    This is used during mitosis to split a carrier's bonds asymmetrically
    based on wavelength, not randomly.
    
    Returns:
        cluster_low: [N] mask for oscillators in lower frequency cluster
        cluster_high: [N] mask for oscillators in higher frequency cluster
    """
    device = osc_omegas.device
    
    # Effective weights
    W = P_col * osc_amplitudes * tuning_col  # [N]
    
    # Find oscillators with significant connection
    significant = W > 0.01
    
    if significant.sum() < 2:
        # Can't partition with < 2 oscillators - give all to "low"
        return significant, torch.zeros_like(significant)
    
    # Weighted median frequency
    sig_indices = torch.where(significant)[0]
    sig_omegas = osc_omegas[sig_indices]
    sig_weights = W[sig_indices]
    
    # Sort by frequency
    sorted_indices = torch.argsort(sig_omegas)
    sorted_weights = sig_weights[sorted_indices]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    total = cumsum[-1]
    
    # Find median split point
    median_idx = torch.searchsorted(cumsum, total / 2)
    
    # Create masks
    cluster_low = torch.zeros_like(significant)
    cluster_high = torch.zeros_like(significant)
    
    # Assign to clusters based on frequency relative to median
    median_omega = sig_omegas[sorted_indices[min(median_idx, len(sorted_indices) - 1)]]
    
    for i in sig_indices:
        if osc_omegas[i] <= median_omega:
            cluster_low[i] = True
        else:
            cluster_high[i] = True
    
    return cluster_low, cluster_high


# =============================================================================
# Coherence Statistics (for mitosis detection)
# =============================================================================

def compute_coherence(
    osc_phasors: torch.Tensor,  # [N] complex
    P: torch.Tensor,            # [N, M]
    osc_amplitudes: torch.Tensor  # [N]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute coherence statistics for each carrier (Eq. 21-24).
    
    coh(k) = |u_k| / Σᵢ w_ik
    baseline b_k = √(π/4) · √(Σᵢ w_ik²) / Σᵢ w_ik
    D_k = coh(k) / b_k
    
    Where w_ik = P_ik · A_i
    
    Returns: (coh, baseline, D) tensors of shape [M]
    """
    if P.numel() == 0 or osc_phasors.numel() == 0:
        empty = torch.empty(0, dtype=DTYPE_REAL, device=P.device)
        return empty, empty, empty
    
    # Weights w_ik = P_ik * A_i
    # P: [N, M], A: [N] -> W: [N, M]
    W = P * osc_amplitudes.unsqueeze(1)
    
    # Carrier drive (ungated for coherence)
    u = torch.matmul(P.T.to(DTYPE_COMPLEX), osc_phasors)  # [M]
    
    # Coherence = |u_k| / Σᵢ w_ik
    sum_w = W.sum(dim=0) + 1e-10  # [M]
    coh = torch.abs(u) / sum_w
    
    # Baseline (random phase expectation)
    # b_k = √(π/4) · √(Σᵢ w_ik²) / Σᵢ w_ik
    sum_w_sq = (W ** 2).sum(dim=0)  # [M]
    baseline = math.sqrt(math.pi / 4) * torch.sqrt(sum_w_sq) / sum_w
    
    # Division score
    D = coh / (baseline + 1e-10)
    
    return coh.to(DTYPE_REAL), baseline, D


# =============================================================================
# Engine
# =============================================================================

@dataclass
class EngineEvents:
    """Record of events for analysis."""
    births: list[dict] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)
    mitoses: list[dict] = field(default_factory=list)


class ResonantEngine:
    """
    The core resonant compression engine.
    
    Implements the full dynamical system from the paper.
    Accepts external signals via step() method.
    """
    
    def __init__(
        self,
        config: Optional[PhysicsConfig] = None,
        device: torch.device = DEVICE,
        seed: int = 0
    ):
        self.config = config or PhysicsConfig()
        self.device = device
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        
        # State
        self.oscillators = OscillatorState(device)
        self.carriers = CarrierState(device)
        self.P = PresenceMatrix(device)
        
        # Time
        self.t = 0.0
        
        # Events
        self.events = EngineEvents()
    
    def add_signal(self, signal: Signal) -> int:
        """
        Add an input signal to the system.
        
        Returns the oscillator ID.
        """
        oid = self.oscillators.add(signal)
        
        # Expand P matrix
        if self.carriers.m > 0:
            self.P.add_oscillator()
        
        return oid
    
    def _nucleate_carrier(self, from_osc_idx: int) -> int:
        """
        Nucleate a new carrier from an unbound oscillator.
        
        The carrier is born aligned to the oscillator (apex principle).
        """
        phase = float(self.oscillators.phases[from_osc_idx])
        freq_hz = float(self.oscillators.omegas[from_osc_idx]) / (2 * math.pi)
        
        # Add carrier
        cid = self.carriers.add(phase, freq_hz, self.t)
        
        # Expand P matrix
        if self.oscillators.n > 0:
            if self.P.P.numel() == 0:
                # First carrier - create matrix
                self.P.P = torch.zeros(
                    self.oscillators.n, 1,
                    dtype=DTYPE_REAL, device=self.device
                )
            else:
                self.P.add_carrier()
        
        # Seed initial bond with triggering oscillator
        self.P.seed_bond(from_osc_idx, self.carriers.m - 1, 0.5)
        
        self.events.births.append({
            "t": self.t,
            "carrier_id": cid,
            "from_osc_idx": from_osc_idx,
            "phase": phase,
            "freq_hz": freq_hz,
        })
        
        return cid
    
    def _check_genesis(self) -> None:
        """
        Check for genesis condition: unbound oscillators with COHERENT high energy.
        
        Genesis requires BOTH:
        1. Sufficient unbound energy (summed amplitude > threshold)
        2. Phase coherence among unbound oscillators (R_unbound > threshold)
        
        This prevents genesis from incoherent noise - only organized
        unbound energy can nucleate a carrier. This makes genesis
        genuinely resonance-driven.
        
        An oscillator is unbound if max_k P_ik < threshold.
        """
        if self.oscillators.n == 0:
            return
        
        # Need at least one oscillator with non-zero amplitude
        if self.oscillators.amplitudes.numel() == 0:
            return
        
        cfg = self.config
        
        if self.carriers.m == 0:
            # No carriers exist - first oscillator with sufficient amplitude nucleates
            # No coherence requirement for the very first carrier
            high_amp = self.oscillators.amplitudes > 0.3
            if high_amp.any():
                # Nucleate from highest amplitude oscillator
                idx = int(self.oscillators.amplitudes.argmax())
                # Double-check index is valid
                if idx < self.oscillators.n:
                    self._nucleate_carrier(idx)
            return
        
        # Find unbound oscillators
        if self.P.P.numel() > 0 and self.P.P.shape[0] == self.oscillators.n:
            max_bonds = self.P.P.max(dim=1).values
        else:
            max_bonds = torch.zeros(self.oscillators.n, device=self.device)
        
        unbound_mask = max_bonds < cfg.unbound_threshold
        
        if not unbound_mask.any():
            return
        
        # Check amplitude pressure
        unbound_amp = self.oscillators.amplitudes[unbound_mask]
        pressure = unbound_amp.sum()
        
        if pressure <= cfg.genesis_pressure:
            return  # Not enough energy
        
        # NEW: Check phase coherence among unbound oscillators
        # Genesis requires organized energy, not just scattered amplitude
        unbound_phases = self.oscillators.phases[unbound_mask]
        
        if unbound_phases.numel() >= 2:
            # Compute order parameter R for unbound oscillators
            # R = |mean(e^(iφ))| - measures phase alignment
            # R = 1: perfect alignment, R ≈ 0: random phases
            phasors = torch.exp(1j * unbound_phases.to(DTYPE_COMPLEX))
            weights = unbound_amp / (unbound_amp.sum() + 1e-8)  # Weight by amplitude
            weighted_phasor = (weights.to(DTYPE_COMPLEX) * phasors).sum()
            R_unbound = float(torch.abs(weighted_phasor))
            
            if R_unbound < cfg.genesis_coherence_threshold:
                return  # Unbound oscillators are not phase-aligned - don't nucleate
        
        # Genesis condition met: sufficient energy AND phase coherence
        # Find strongest unbound oscillator to nucleate from
        unbound_indices = torch.where(unbound_mask)[0]
        strongest_idx = unbound_indices[unbound_amp.argmax()]
        if strongest_idx < self.oscillators.n:
            self._nucleate_carrier(int(strongest_idx))
    
    def _check_mitosis(self) -> None:
        """
        Check for mitosis condition: carriers with persistent spectral conflict.
        
        Mitosis is SPECTRAL FACTORIZATION, not blind duplication.
        A carrier divides when:
        1. Its bonded oscillators form ≥2 distinct frequency clusters
        2. Coherence is persistently low (D < threshold)
        3. This persists over the mitosis window
        
        Division produces immediately differentiated offspring, not duplicates.
        """
        if self.carriers.m == 0 or self.oscillators.n < 2:
            return
        
        cfg = self.config
        
        # Compute coherence
        _, _, D = compute_coherence(
            self.oscillators.phasors,
            self.P.P,
            self.oscillators.amplitudes
        )
        
        # Compute tuning for spectral analysis
        tuning = compute_tuning_strength(
            self.carriers.phases,
            self.oscillators.phases,
            cfg.gate_width
        )
        
        # Compute spectral profiles
        _, spectral_var, is_multimodal = compute_spectral_profiles(
            self.oscillators.omegas,
            self.oscillators.amplitudes,
            self.P.P,
            tuning
        )
        
        # Update coherence EMA
        if D.numel() > 0:
            alpha = cfg.dt / cfg.intake_tau
            self.carriers.coherence_ema = (1 - alpha) * self.carriers.coherence_ema + alpha * D
        
        # Division condition: low coherence AND multimodal spectrum
        # This ensures we only divide when there's actual spectral conflict
        below = (D < cfg.d_threshold) & is_multimodal
        self.carriers.d_below_threshold = torch.where(
            below,
            self.carriers.d_below_threshold + 1,
            torch.zeros_like(self.carriers.d_below_threshold)
        )
        
        # Trigger mitosis for carriers that have been unstable long enough
        mitosis_mask = self.carriers.d_below_threshold >= cfg.mitosis_window
        
        for idx in torch.where(mitosis_mask)[0].tolist():
            self._do_spectral_mitosis(idx, tuning)
            # Reset counter
            self.carriers.d_below_threshold[idx] = 0
    
    def _do_spectral_mitosis(self, carrier_idx: int, tuning: torch.Tensor) -> None:
        """
        Perform SPECTRAL FACTORIZATION: partition bonds by wavelength.
        
        This is NOT blind duplication. It is wavelength-aware bond partitioning:
        1. Cluster bonded oscillators by frequency
        2. Parent keeps one cluster, child gets the other
        3. Each offspring's ω is shifted to its cluster's spectral center
        
        This produces IMMEDIATELY differentiated carriers, not duplicates.
        """
        # Get parent's bonds and tuning
        P_col = self.P.P[:, carrier_idx]
        tuning_col = tuning[:, carrier_idx]
        
        # Partition oscillators by frequency
        cluster_low, cluster_high = partition_oscillators_by_frequency(
            self.oscillators.omegas,
            P_col,
            tuning_col,
            self.oscillators.amplitudes
        )
        
        # Check if partition is meaningful (both clusters have oscillators)
        if cluster_low.sum() == 0 or cluster_high.sum() == 0:
            # Can't partition meaningfully - skip mitosis
            return
        
        # Compute spectral centers for each cluster
        W = P_col * self.oscillators.amplitudes * tuning_col
        
        # Low cluster center
        W_low = W * cluster_low.float()
        if W_low.sum() > 0:
            omega_low = (W_low * self.oscillators.omegas).sum() / W_low.sum()
        else:
            omega_low = self.carriers.omegas[carrier_idx]
        
        # High cluster center
        W_high = W * cluster_high.float()
        if W_high.sum() > 0:
            omega_high = (W_high * self.oscillators.omegas).sum() / W_high.sum()
        else:
            omega_high = self.carriers.omegas[carrier_idx]
        
        # Parent keeps LOW cluster, shifts frequency
        self.carriers.omegas[carrier_idx] = omega_low
        
        # Zero out parent's bonds to high cluster oscillators
        self.P.P[cluster_high, carrier_idx] = 0.0
        
        # Create child with HIGH cluster
        parent_c = self.carriers.c[carrier_idx]
        phase = float(torch.angle(parent_c))
        freq_hz_high = float(omega_high) / (2 * math.pi)
        parent_gate_width = float(self.carriers.gate_widths[carrier_idx])
        
        cid = self.carriers.add(phase, freq_hz_high, self.t, gate_width=parent_gate_width)
        
        # Expand P and give child only the HIGH cluster bonds
        if self.P.P.numel() > 0:
            self.P.add_carrier()
            # Child gets only the high cluster bonds
            self.P.P[:, -1] = P_col * cluster_high.float()
        
        # Apply small noise to differentiate phases
        noise_std = 0.05  # Smaller than before - differentiation is primarily spectral
        noise_parent = noise_std * (torch.randn(2, generator=self.rng).to(self.device))
        noise_child = noise_std * (torch.randn(2, generator=self.rng).to(self.device))
        
        self.carriers.c[carrier_idx] += (noise_parent[0] + 1j * noise_parent[1])
        self.carriers.c[-1] += (noise_child[0] + 1j * noise_child[1])
        
        self.events.mitoses.append({
            "t": self.t,
            "parent_idx": carrier_idx,
            "child_id": cid,
            "parent_omega_hz": float(omega_low) / (2 * math.pi),
            "child_omega_hz": freq_hz_high,
            "low_cluster_size": int(cluster_low.sum()),
            "high_cluster_size": int(cluster_high.sum()),
        })
    
    def _check_dissolution(self) -> None:
        """
        Check for dissolution: carriers that fail to capture sufficient COHERENT energy.
        
        IMPORTANT: Metabolism is now COHERENCE-WEIGHTED.
        A carrier that captures incoherent noise does not "eat well" even at high amplitude.
        This aligns survival with representational quality.
        
        Effective intake = raw intake × coherence
        """
        if self.carriers.m == 0:
            return
        
        cfg = self.config
        
        # Effective intake is coherence-weighted
        # Coherent energy is "nutritious", incoherent energy is not
        effective_intake = self.carriers.intake_ema * self.carriers.coherence_ema
        
        # Check starving carriers
        starving = effective_intake < cfg.metabolic_cost
        self.carriers.starve_steps = torch.where(
            starving,
            self.carriers.starve_steps + 1,
            torch.zeros_like(self.carriers.starve_steps)
        )
        
        # Dissolve carriers that have starved too long
        dissolve_mask = self.carriers.starve_steps >= cfg.starve_window
        
        if dissolve_mask.any():
            indices = torch.where(dissolve_mask)[0]
            death_events = self.carriers.remove(indices)
            self.P.remove_carriers(indices)
            
            for event in death_events:
                event["t"] = self.t
                event["reason"] = "starvation"
                self.events.deaths.append(event)
    
    def _update_gate_widths(self) -> None:
        """
        Update gate widths based on coherence history.
        
        This creates EMERGENT SPECIALIZATION:
        - Sustained high coherence → gate narrows (more selective, specialized)
        - Persistent low coherence → gate widens (more exploratory)
        
        Gate width reflects perceptual tolerance. Narrow-gate carriers are
        specialists that lock onto specific wavelengths. Wide-gate carriers
        are generalists still searching for their spectral niche.
        
        dW/dt = -η_narrow * (D - 1) * W     (narrows when D > 1)
              + η_widen * (1 - D) * W        (widens when D < 1)
        
        Simplified: dW/dt = rate * (1 - D) * W
        where rate > 0 means widen, rate < 0 means narrow
        """
        if self.carriers.m == 0:
            return
        
        cfg = self.config
        dt = cfg.dt
        
        # Use coherence EMA for stability
        D = self.carriers.coherence_ema
        
        # Compute gate width change
        # D > 1: narrowing (high coherence → specialization)
        # D < 1: widening (low coherence → exploration)
        dW_narrow = -cfg.gate_narrow_rate * (D - 1.0) * self.carriers.gate_widths
        dW_widen = cfg.gate_widen_rate * (1.0 - D) * self.carriers.gate_widths
        
        # Total change: narrowing when coherent, widening when incoherent
        dW = torch.where(D > 1.0, dW_narrow, dW_widen)
        
        # Update and clamp to valid range
        self.carriers.gate_widths = torch.clamp(
            self.carriers.gate_widths + dW * dt,
            min=cfg.gate_width_min,
            max=cfg.gate_width_max
        )
    
    def _update_bonds(self) -> None:
        """
        Update elastic bonds P based on resonant capture.
        
        Bonds strengthen when:
        1. Gate is open (G = 1)
        2. Tuning is strong (T close to 1) - this is the antenna principle!
        3. Energy is being captured (Re(c·e^(-iφ)) > 0)
        
        Bonds decay continuously and snap below threshold.
        
        The key insight: reinforcement is modulated by TUNING STRENGTH,
        not just a constant. Well-aligned oscillators form stronger bonds.
        """
        if self.P.P.numel() == 0 or self.oscillators.n == 0 or self.carriers.m == 0:
            return
        
        cfg = self.config
        dt = cfg.dt
        
        # Gate values [M]
        gates = self.carriers.gate()
        
        # Tuning strength [N, M] - the antenna principle!
        tuning = compute_tuning_strength(
            self.carriers.phases,
            self.oscillators.phases,
            cfg.gate_width
        )
        
        # Energy capture indicator: Re(c_k · e^(-iφ_i))
        # c: [M], φ: [N] -> capture: [N, M]
        c_expanded = self.carriers.c.unsqueeze(0)  # [1, M]
        phi_expanded = self.oscillators.phases.unsqueeze(1)  # [N, 1]
        capture = torch.real(c_expanded * torch.exp(-1j * phi_expanded.to(DTYPE_COMPLEX)))
        
        # Reinforcement: gate × tuning × positive capture
        # Tuning modulates how much reinforcement happens - better alignment = faster learning
        reinforcement = (
            cfg.alpha_p * 
            gates.unsqueeze(0) *      # Only when gate is open
            tuning *                   # Modulated by tuning strength (antenna principle)
            self.P.P *                 # Proportional to current bond
            torch.clamp(capture, min=0.0)  # Only positive capture
        )
        
        # Decay - bonds must be continuously sustained
        decay = cfg.lambda_p * self.P.P
        
        # Update P
        dP = (reinforcement - decay) / cfg.tau_p
        self.P.P = torch.clamp(self.P.P + dP * dt, min=0.0, max=1.0)
        
        # Snap bonds below threshold
        self.P.P = torch.where(
            self.P.P < cfg.p_snap,
            torch.zeros_like(self.P.P),
            self.P.P
        )
    
    def _update_carriers(self, drive: torch.Tensor) -> None:
        """
        Update carrier dynamics (Eq. 10).
        
        dc/dt = u - γc - β|c|²c
        """
        if self.carriers.m == 0:
            return
        
        cfg = self.config
        dt = cfg.dt
        
        # Dynamics
        damping = cfg.gamma * self.carriers.c
        saturation = cfg.beta * (torch.abs(self.carriers.c) ** 2) * self.carriers.c
        
        dc = drive - damping - saturation
        self.carriers.c = self.carriers.c + dc * dt
        
        # Update intake EMA
        intake = torch.abs(drive)
        alpha_ema = dt / cfg.intake_tau
        self.carriers.intake_ema = (1 - alpha_ema) * self.carriers.intake_ema + alpha_ema * intake
    
    def _update_oscillators(self, phase_influence: torch.Tensor) -> None:
        """
        Update oscillator dynamics (Eq. 12 and amplitude).
        
        dφ/dt = ω + phase_influence
        """
        if self.oscillators.n == 0:
            return
        
        cfg = self.config
        dt = cfg.dt
        
        # Phase evolution
        dphase = self.oscillators.omegas + phase_influence
        self.oscillators.phases = (self.oscillators.phases + dphase * dt) % (2 * math.pi)
        
        # Amplitude evolution
        self.oscillators.step_amplitudes(dt)
    
    def _garbage_collect(self) -> None:
        """Remove dead oscillators (below noise floor and not driven)."""
        if self.oscillators.n == 0:
            return
        
        cfg = self.config
        
        dead = (self.oscillators.amplitudes < cfg.noise_floor) & (self.oscillators.drive_remaining <= 0)
        
        if dead.any():
            indices = torch.where(dead)[0]
            self.oscillators.remove(indices)
            self.P.remove_oscillators(indices)
    
    def step(self, signals: Optional[list[Signal]] = None) -> None:
        """
        Advance the simulation by one time step.
        
        Args:
            signals: Optional list of new signals to add this step.
        """
        cfg = self.config
        
        # 1. Add new signals
        if signals:
            for sig in signals:
                self.add_signal(sig)
        
        # 2. Check genesis (nucleate carriers for unbound oscillators)
        self._check_genesis()
        
        # 3. Compute tuning strength (THE ANTENNA PRINCIPLE)
        # Now uses PER-CARRIER gate widths for emergent specialization
        if self.carriers.m > 0 and self.oscillators.n > 0:
            tuning = compute_tuning_strength_per_carrier(
                self.carriers.phases,
                self.oscillators.phases,
                self.carriers.gate_widths
            )
            gates = self.carriers.gate()
            
            # 4. Compute carrier drive with tuning-weighted coupling
            drive = compute_carrier_drive(
                self.oscillators.phasors,
                self.P.P,
                gates,
                tuning
            )
        else:
            tuning = torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device)
            drive = torch.empty(0, dtype=DTYPE_COMPLEX, device=self.device)
        
        # 5. Update carriers
        self._update_carriers(drive)
        
        # 6. Compute back-influence with tuning
        if self.carriers.m > 0 and self.oscillators.n > 0 and self.P.P.numel() > 0:
            g = compute_back_influence(self.carriers.c, self.P.P, tuning)
            phase_influence = compute_phase_influence(g, self.oscillators.phases)
        else:
            # No coupling - just zeros for each oscillator
            phase_influence = torch.zeros(self.oscillators.n, dtype=DTYPE_REAL, device=self.device)
        
        # 7. Update oscillators
        self._update_oscillators(phase_influence)
        
        # 8. Update bonds
        self._update_bonds()
        
        # 9. Check mitosis (spectral factorization)
        self._check_mitosis()
        
        # 10. Update gate widths (emergent specialization)
        self._update_gate_widths()
        
        # 11. Check dissolution (coherence-weighted metabolism)
        self._check_dissolution()
        
        # 12. Garbage collect
        self._garbage_collect()
        
        # Advance time
        self.t += cfg.dt
    
    # =========================================================================
    # Metrics
    # =========================================================================
    
    def nnz_P(self) -> int:
        """Number of non-zero bonds."""
        return self.P.nnz()
    
    def global_sync_R(self) -> float:
        """Global synchronization order parameter (Eq. 32)."""
        if self.oscillators.n == 0:
            return 0.0
        
        active = self.oscillators.amplitudes > self.config.noise_floor
        if not active.any():
            return 0.0
        
        phasors = torch.exp(1j * self.oscillators.phases[active].to(DTYPE_COMPLEX))
        R = torch.abs(phasors.mean())
        return float(R)
    
    def L_comp(self) -> int:
        """Description length proxy (Eq. 34)."""
        return self.nnz_P() + self.carriers.m
    
    # =========================================================================
    # Observation Interface (CANONICAL - supports classification & prediction)
    # =========================================================================
    
    def observe(self) -> dict:
        """
        Canonical observation interface.
        
        This is the ONLY interface needed for external tasks.
        The same observation supports:
        - Classification (what carrier configurations exist)
        - Next-token prediction (how state evolves)
        - Unsupervised structure discovery
        - Hybrid tasks
        
        CRITICAL DESIGN RULE:
        Outputs MEASURE resonance, they do not PARTICIPATE in it.
        No gradients, no reward shaping, no feedback into dynamics.
        Tasks are consumers, not drivers.
        
        Returns a state snapshot containing all observables needed for
        downstream interpretation, without modifying engine state.
        """
        # Compute tuning matrix for soft assignments
        if self.carriers.m > 0 and self.oscillators.n > 0:
            tuning = compute_tuning_strength_per_carrier(
                self.carriers.phases,
                self.oscillators.phases,
                self.carriers.gate_widths
            )
            
            # Compute spectral profiles
            omega_center, spectral_var, is_multimodal = compute_spectral_profiles(
                self.oscillators.omegas,
                self.oscillators.amplitudes,
                self.P.P,
                tuning
            )
            
            # Soft assignment matrix: m_ik ∝ P_ik · T_ik (normalized over k)
            raw_assignment = self.P.P * tuning
            assignment_sum = raw_assignment.sum(dim=1, keepdim=True) + 1e-8
            soft_assignment = raw_assignment / assignment_sum
        else:
            tuning = torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device)
            omega_center = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            spectral_var = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            is_multimodal = torch.empty(0, dtype=torch.bool, device=self.device)
            soft_assignment = torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device)
        
        return {
            # Time
            "t": self.t,
            
            # Population counts
            "n_oscillators": self.oscillators.n,
            "n_carriers": self.carriers.m,
            "n_bonds": self.nnz_P(),
            
            # Global metrics
            "global_sync_R": self.global_sync_R(),
            "L_comp": self.L_comp(),
            
            # Carrier-level observables [M]
            "carrier_energy": self.carriers.energies.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_phase": self.carriers.phases.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_omega": self.carriers.omegas.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_intake": self.carriers.intake_ema.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_coherence": self.carriers.coherence_ema.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_gate_width": self.carriers.gate_widths.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_spectral_center": omega_center,
            "carrier_spectral_variance": spectral_var,
            "carrier_is_multimodal": is_multimodal,
            
            # Oscillator-level observables [N]
            "osc_amplitude": self.oscillators.amplitudes.clone() if self.oscillators.n > 0 else torch.empty(0, device=self.device),
            "osc_phase": self.oscillators.phases.clone() if self.oscillators.n > 0 else torch.empty(0, device=self.device),
            "osc_omega": self.oscillators.omegas.clone() if self.oscillators.n > 0 else torch.empty(0, device=self.device),
            
            # Soft assignment matrix [N, M]
            # m_ik = how strongly oscillator i is assigned to carrier k
            # Normalized over k so each oscillator's assignments sum to 1
            "soft_assignment": soft_assignment,
            
            # Tuning matrix [N, M] - raw coupling strengths
            "tuning": tuning,
            
            # Presence matrix [N, M] - bond strengths
            "presence": self.P.P.clone() if self.P.P.numel() > 0 else torch.empty(0, 0, device=self.device),
        }
    
    def observe_carriers_only(self) -> torch.Tensor:
        """
        Compact carrier state vector for sequence modeling.
        
        Returns a [M, D] tensor where D is the feature dimension per carrier.
        Suitable for feeding into transformers, RNNs, or Markov models.
        
        Features per carrier:
        - energy (1)
        - phase (2 - sin/cos encoding)
        - omega (1)
        - intake (1)
        - coherence (1)
        - gate_width (1)
        - spectral_center (1)
        - spectral_variance (1)
        
        Total: D = 9 features per carrier
        """
        if self.carriers.m == 0:
            return torch.empty(0, 9, dtype=DTYPE_REAL, device=self.device)
        
        obs = self.observe()
        
        features = torch.stack([
            obs["carrier_energy"],
            torch.sin(obs["carrier_phase"]),
            torch.cos(obs["carrier_phase"]),
            obs["carrier_omega"] / (2 * math.pi),  # Normalize to Hz
            obs["carrier_intake"],
            obs["carrier_coherence"],
            obs["carrier_gate_width"] / math.pi,  # Normalize to units of π
            obs["carrier_spectral_center"] / (2 * math.pi),  # Normalize to Hz
            torch.sqrt(obs["carrier_spectral_variance"] + 1e-8) / (2 * math.pi),  # Std in Hz
        ], dim=1)
        
        return features
    
    def observe_global(self) -> torch.Tensor:
        """
        Global state summary vector.
        
        Returns a [D] tensor summarizing the entire system state.
        Suitable for classification tasks where input identity matters less
        than overall configuration.
        
        Features:
        - n_carriers (1)
        - n_oscillators (1)
        - global_sync_R (1)
        - L_comp (1)
        - mean carrier energy (1)
        - mean carrier coherence (1)
        - carrier energy entropy (1) - how distributed is energy
        - spectral coverage (1) - range of represented frequencies
        
        Total: D = 8 features
        """
        obs = self.observe()
        
        # Compute derived statistics
        if obs["n_carriers"] > 0:
            energies = obs["carrier_energy"]
            mean_energy = energies.mean()
            mean_coherence = obs["carrier_coherence"].mean()
            
            # Energy entropy (how distributed)
            energy_probs = energies / (energies.sum() + 1e-8)
            energy_entropy = -(energy_probs * torch.log(energy_probs + 1e-8)).sum()
            
            # Spectral coverage (range of represented frequencies)
            spectral_centers = obs["carrier_spectral_center"]
            spectral_coverage = (spectral_centers.max() - spectral_centers.min()) / (2 * math.pi)
        else:
            mean_energy = torch.tensor(0.0, device=self.device)
            mean_coherence = torch.tensor(0.0, device=self.device)
            energy_entropy = torch.tensor(0.0, device=self.device)
            spectral_coverage = torch.tensor(0.0, device=self.device)
        
        return torch.tensor([
            float(obs["n_carriers"]),
            float(obs["n_oscillators"]),
            obs["global_sync_R"],
            float(obs["L_comp"]),
            float(mean_energy),
            float(mean_coherence),
            float(energy_entropy),
            float(spectral_coverage),
        ], dtype=DTYPE_REAL, device=self.device)


# =============================================================================
# Simple Signal Source for Testing
# =============================================================================

class StochasticStream(SignalSource):
    """Stochastic stream of signals for testing."""
    
    def __init__(
        self,
        event_rate_hz: float = 0.9,
        freq_range: tuple[float, float] = (0.6, 4.5),
        duration_range: tuple[float, float] = (0.8, 2.8),
        seed: int = 0
    ):
        self.event_rate_hz = event_rate_hz
        self.freq_range = freq_range
        self.duration_range = duration_range
        self.rng = torch.Generator().manual_seed(seed)
    
    def get_signals(self, t: float, dt: float) -> list[Signal]:
        """Generate signals for this time step."""
        lam = self.event_rate_hz * dt
        
        # Poisson sampling
        if lam >= 0.25:
            n = int(torch.poisson(torch.tensor([lam]), generator=self.rng).item())
        else:
            n = 1 if torch.rand(1, generator=self.rng).item() < lam else 0
        
        signals = []
        for _ in range(n):
            freq = self.freq_range[0] + torch.rand(1, generator=self.rng).item() * (self.freq_range[1] - self.freq_range[0])
            dur = self.duration_range[0] + torch.rand(1, generator=self.rng).item() * (self.duration_range[1] - self.duration_range[0])
            phase = torch.rand(1, generator=self.rng).item() * 2 * math.pi
            
            signals.append(Signal(
                freq_hz=freq,
                phase=phase,
                amplitude=1.0,
                duration_s=dur
            ))
        
        return signals


# =============================================================================
# Main (for quick testing)
# =============================================================================

def main():
    """Quick test of the engine."""
    print("=" * 60)
    print("Resonant Compression Systems — Engine Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    engine = ResonantEngine(seed=0)
    stream = StochasticStream(seed=0)
    
    # Run for 10 seconds
    dt = engine.config.dt
    steps = int(10.0 / dt)
    
    for step in range(steps):
        signals = stream.get_signals(engine.t, dt)
        engine.step(signals)
        
        if step % 200 == 0:
            print(f"t={engine.t:6.2f}s | N={engine.oscillators.n:3d} | M={engine.carriers.m:2d} | "
                  f"nnz={engine.nnz_P():4d} | R={engine.global_sync_R():.3f}")
    
    print()
    print(f"Final: {engine.oscillators.n} oscillators, {engine.carriers.m} carriers")
    print(f"Births: {len(engine.events.births)}, Deaths: {len(engine.events.deaths)}, Mitoses: {len(engine.events.mitoses)}")


if __name__ == "__main__":
    main()
