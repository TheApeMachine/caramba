
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_REAL = torch.float32
DTYPE_COMPLEX = torch.complex64


# =============================================================================
# Physics Constants (from paper, Section 4)
# =============================================================================
#
# CHANGE (2026-01-30):
# -------------------
# The original implementation treated the values in PhysicsConfig as *static*
# constants. This makes the model look "controlled" by hand-tuned knobs, and
# it also contradicts the physical intuition that material properties (damping,
# saturation, bond mobility, metabolic thresholds, etc.) are *state-dependent*.
#
# In real media, quantities like viscosity, diffusion coefficients, and
# effective damping are not immutable: they vary with temperature, density,
# load, and ongoing energy flux.
#
# The code below keeps the paper values as *reference / equilibrium* values
# (i.e., what the medium looks like around a nominal operating point), but the
# engine no longer uses them directly. Instead, an internal "MediumState"
# evolves from the system's own dynamics, and an InstantPhysics snapshot is
# derived from that medium state every step.
#
# Key design constraint:
# - No gradients, no backprop, no external controller.
# - The medium updates only from observables already produced by the dynamics.
# - The paper constants are treated as equilibrium scales, not tuning knobs.
#
# This is the minimal move that converts "hard-coded constants" into
# "emergent state-dependent material properties", while preserving the original
# paper-aligned behavior when the medium sits near its nominal equilibrium.
# =============================================================================


@dataclass
class PhysicsConfig:
    """
    Reference (equilibrium) material constants of the resonant system.

    IMPORTANT (new interpretation):
    - The values in this dataclass are NOT assumed to be globally constant.
    - They are interpreted as the medium's *nominal* properties at
      temperature ~= 1 and resource ~= 1.
    - If emergent_physics=True, the engine derives instantaneous effective
      values each step from MediumState.

    This preserves the paper numbers as an anchor, but lets the system
    discover its own fluctuating equilibrium.
    """

    # =========================================================================
    # SIMULATION (DERIVED FROM SIGNAL FREQUENCIES)
    # =========================================================================
    dt: float = 0.005

    # =========================================================================
    # SIGNAL INGESTION (INPUT -> THERMODYNAMICS + BOND SEEDING)
    # =========================================================================
    input_energy_scale: float = 0.2
    input_bond_threshold: float = 0.7
    input_bond_strength: float = 0.25
    input_max_bonds_per_signal: int = 3
    force_genesis_on_input: bool = False
    carrier_grace_steps: int = 0
    genesis_from_first_signal_only: bool = False

    # =========================================================================
    # CARRIER GATE (DERIVED FROM GATE GEOMETRY)
    # =========================================================================
    gate_width: float = math.pi

    # =========================================================================
    # CARRIER PHYSICS (Eq. 10-11) -- reference values
    # =========================================================================
    gamma: float = 2.0
    beta: float = 0.5

    # =========================================================================
    # ELASTIC BONDS (Eq. 19) -- reference values
    # =========================================================================
    tau_p: float = 6.0
    alpha_p: float = 2.0
    lambda_p: float = 0.25
    p_snap: float = 0.01

    # Backwards-compatible alias
    @property
    def snap_threshold(self) -> float:
        return float(self.p_snap)

    @snap_threshold.setter
    def snap_threshold(self, value: float) -> None:
        self.p_snap = float(value)

    # =========================================================================
    # COHERENCE AND MITOSIS (Eq. 24-25) -- reference values
    # =========================================================================
    d_threshold: float = 0.95
    mitosis_window: int = 50

    # =========================================================================
    # METABOLISM -- reference values
    # =========================================================================
    intake_tau: float = 2.5
    metabolic_cost: float = 0.2
    starve_window: int = 300
    coherence_tau: float = 1.0

    # =========================================================================
    # GENESIS -- reference values
    # =========================================================================
    unbound_threshold: float = 0.1
    genesis_pressure: float = 1.25
    genesis_coherence_threshold: float = 0.5

    # =========================================================================
    # GATE WIDTH EVOLUTION (Specialization) -- reference values
    # =========================================================================
    gate_width_min: float = math.pi / 4
    gate_width_max: float = math.pi * 1.5
    gate_narrow_rate: float = 0.1
    gate_widen_rate: float = 0.05

    # =========================================================================
    # TOP-DOWN FEEDBACK (Observation-driven utility shaping)
    # =========================================================================
    top_down_tau: float = 0.5
    top_down_gate_rate: float = 0.15
    top_down_noise_penalty: float = 0.5
    top_down_max_abs: float = 1.0

    # =========================================================================
    # NOISE FLOOR
    # =========================================================================
    noise_floor: float = 1e-3

    # =========================================================================
    # OSCILLATOR AMPLITUDE PHYSICS (Eq. 17) -- reference values
    # =========================================================================
    # CHANGE:
    # The previous code hard-coded oscillator amplitude damping=1.5 and
    # saturation=0.2 inside OscillatorState.step_amplitudes(). These are
    # clearly "physics-like constants" and should be subject to the same
    # emergent-medium mechanism as gamma/beta/etc.
    #
    # These are therefore promoted into PhysicsConfig as equilibrium values.
    osc_amp_damping: float = 1.5
    osc_amp_saturation: float = 0.2

    # =========================================================================
    # EMERGENT MEDIUM (NEW)
    # =========================================================================
    # If True, the engine derives effective physics values from MediumState.
    # If False, the engine behaves like the original implementation.
    emergent_physics: bool = True

    # The medium state uses very conservative bounds to avoid numerical
    # blow-ups. These are not intended as "tuning"; they are safety rails
    # on unobserved latent variables (like preventing negative temperature).
    medium_temperature_min: float = 0.5
    medium_temperature_max: float = 3.0
    medium_resource_min: float = 0.05
    medium_resource_max: float = 1.0


# =============================================================================
# Emergent Medium -> Instantaneous Physics (NEW)
# =============================================================================

@dataclass
class InstantPhysics:
    """
    Effective physics parameters for a *single* integration step.

    This is what the engine actually uses internally when emergent_physics=True.
    Values are derived from MediumState + reference scales in PhysicsConfig.
    """
    # Simulation
    dt: float

    # Medium scalars (observable for debugging / analysis)
    temperature: float
    resource: float

    # Carrier dynamics
    gamma: float
    beta: float

    # Bond dynamics
    tau_p: float
    alpha_p: float
    lambda_p: float
    p_snap: float

    # Mitosis / coherence
    d_threshold: float
    mitosis_window: int
    coherence_tau: float

    # Metabolism
    intake_tau: float
    metabolic_cost: float
    starve_window: int

    # Genesis
    unbound_threshold: float
    genesis_pressure: float
    genesis_coherence_threshold: float

    # Gate width adaptation rates
    gate_narrow_rate: float
    gate_widen_rate: float

    # Oscillator amplitude dynamics
    osc_amp_damping: float
    osc_amp_saturation: float

    @staticmethod
    def from_config(cfg: PhysicsConfig) -> "InstantPhysics":
        """Degenerate 'no emergence' mode: just mirror the reference constants."""
        return InstantPhysics(
            dt=float(cfg.dt),
            temperature=1.0,
            resource=1.0,
            gamma=float(cfg.gamma),
            beta=float(cfg.beta),
            tau_p=float(cfg.tau_p),
            alpha_p=float(cfg.alpha_p),
            lambda_p=float(cfg.lambda_p),
            p_snap=float(cfg.p_snap),
            d_threshold=float(cfg.d_threshold),
            mitosis_window=int(cfg.mitosis_window),
            coherence_tau=float(cfg.coherence_tau),
            intake_tau=float(cfg.intake_tau),
            metabolic_cost=float(cfg.metabolic_cost),
            starve_window=int(cfg.starve_window),
            unbound_threshold=float(cfg.unbound_threshold),
            genesis_pressure=float(cfg.genesis_pressure),
            genesis_coherence_threshold=float(cfg.genesis_coherence_threshold),
            gate_narrow_rate=float(cfg.gate_narrow_rate),
            gate_widen_rate=float(cfg.gate_widen_rate),
            osc_amp_damping=float(cfg.osc_amp_damping),
            osc_amp_saturation=float(cfg.osc_amp_saturation),
        )


class MediumState:
    """
    Internal medium state that makes "constants" emergent.

    We intentionally keep the latent state extremely low-dimensional:
      - temperature (T): disorder / noise / effective viscosity proxy
      - resource    (R): available free energy / "nutrient density" proxy

    Both variables evolve *only* from system observables (supply, demand, coherence).
    No gradients; no external controller; no tuning during runtime.
    """

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.temperature = torch.tensor(1.0, dtype=DTYPE_REAL, device=device)
        self.resource = torch.tensor(1.0, dtype=DTYPE_REAL, device=device)

    def _clamp_medium(self, cfg: PhysicsConfig) -> None:
        self.temperature = torch.clamp(
            self.temperature,
            min=float(cfg.medium_temperature_min),
            max=float(cfg.medium_temperature_max),
        )
        self.resource = torch.clamp(
            self.resource,
            min=float(cfg.medium_resource_min),
            max=float(cfg.medium_resource_max),
        )

    def update(
        self,
        cfg: PhysicsConfig,
        *,
        supply: torch.Tensor,          # scalar >= 0  (oscillator energy available)
        demand: torch.Tensor,          # scalar >= 0  (carrier "consumption"/intake)
        coherent_demand: torch.Tensor, # scalar >= 0  (coherent share of intake)
        carrier_energy: torch.Tensor,  # scalar >= 0  (crowding proxy)
    ) -> None:
        """
        Evolve the medium state one step.

        CHANGE:
        - This update is the *mechanism* by which material constants become emergent.
        - It is deliberately simple and uses only ratios/bounded terms so it does
          not introduce new free scales.

        Resource target:
            R_target = supply / (supply + demand)
        Temperature target:
            incoh_frac = (demand - coherent_demand) / demand
            crowd      = carrier_energy / (carrier_energy + 1)
            T_target   = 1 + incoh_frac * (1 + crowd)

        Both targets are filtered with EMAs whose time constants are taken from
        the reference physics itself:
            - resource relaxes on ~intake_tau
            - temperature relaxes on ~tau_p

        This means we did NOT introduce new arbitrary "controller" timescales.
        """
        if not cfg.emergent_physics:
            return

        dt = float(cfg.dt)
        eps = 1e-8

        supply = torch.clamp(supply, min=0.0)
        demand = torch.clamp(demand, min=0.0)
        coherent_demand = torch.clamp(coherent_demand, min=0.0)
        carrier_energy = torch.clamp(carrier_energy, min=0.0)

        total = supply + demand
        # If there is essentially no activity, let resource recover.
        R_target = torch.where(total > eps, supply / (total + eps), torch.ones_like(total))
        R_target = torch.clamp(R_target, 0.0, 1.0)

        incoherent = torch.clamp(demand - coherent_demand, min=0.0)
        incoh_frac = torch.where(demand > eps, incoherent / (demand + eps), torch.zeros_like(demand))
        incoh_frac = torch.clamp(incoh_frac, 0.0, 1.0)

        crowd = carrier_energy / (carrier_energy + 1.0)
        crowd = torch.clamp(crowd, 0.0, 1.0)

        T_target = 1.0 + incoh_frac * (1.0 + crowd)
        T_target = torch.clamp(
            T_target,
            min=float(cfg.medium_temperature_min),
            max=float(cfg.medium_temperature_max),
        )

        # EMA updates using reference timescales (no new "magic" taus)
        alpha_R = dt / max(float(cfg.intake_tau), dt)
        alpha_T = dt / max(float(cfg.tau_p), dt)

        alpha_R = float(max(0.0, min(1.0, alpha_R)))
        alpha_T = float(max(0.0, min(1.0, alpha_T)))

        self.resource = (1.0 - alpha_R) * self.resource + alpha_R * R_target
        self.temperature = (1.0 - alpha_T) * self.temperature + alpha_T * T_target

        self._clamp_medium(cfg)

    def instant(self, cfg: PhysicsConfig) -> InstantPhysics:
        """
        Derive the instantaneous effective physics from the current medium state.

        CHANGE:
        - This replaces direct use of cfg.gamma/cfg.beta/... throughout the engine.
        - The reference values are treated as equilibrium scales.
        """
        if not cfg.emergent_physics:
            return InstantPhysics.from_config(cfg)

        # Medium scalars as Python floats (cheap; used as multipliers in torch ops).
        T = float(self.temperature.item())
        R = float(self.resource.item())
        eps = 1e-6

        # Helper: clamp a value to a factor-range of its reference value.
        # This is purely a numerical stability rail, not a "control policy".
        def clamp_mul(val: float, ref: float, lo: float, hi: float) -> float:
            if ref == 0.0:
                return float(val)
            return float(max(ref * lo, min(ref * hi, val)))

        # ------------------------------------------------------------
        # Carrier dynamics
        # ------------------------------------------------------------
        gamma = float(cfg.gamma) * T
        gamma = clamp_mul(gamma, float(cfg.gamma), lo=0.25, hi=4.0)

        beta = float(cfg.beta) / (R + eps)
        beta = clamp_mul(beta, float(cfg.beta), lo=0.25, hi=20.0)

        # ------------------------------------------------------------
        # Bond dynamics
        # ------------------------------------------------------------
        # Intuition:
        # - higher temperature -> bonds more labile (faster changes): tau_p decreases
        # - scarce resources   -> less "glue": alpha_p decreases, lambda_p increases
        tau_p = float(cfg.tau_p) / T
        tau_p = clamp_mul(tau_p, float(cfg.tau_p), lo=0.25, hi=4.0)

        alpha_p = float(cfg.alpha_p) * (R / (T + eps))
        alpha_p = clamp_mul(alpha_p, float(cfg.alpha_p), lo=0.05, hi=4.0)

        lambda_p = float(cfg.lambda_p) * (T / (R + eps))
        lambda_p = clamp_mul(lambda_p, float(cfg.lambda_p), lo=0.25, hi=20.0)

        # Snap threshold remains a geometry/noise-floor concept; keep it fixed.
        p_snap = float(cfg.p_snap)

        # ------------------------------------------------------------
        # Metabolism
        # ------------------------------------------------------------
        intake_tau = float(cfg.intake_tau) / T
        intake_tau = clamp_mul(intake_tau, float(cfg.intake_tau), lo=0.25, hi=4.0)

        coherence_tau = float(cfg.coherence_tau) / T
        coherence_tau = clamp_mul(coherence_tau, float(cfg.coherence_tau), lo=0.25, hi=4.0)

        metabolic_cost = float(cfg.metabolic_cost) * (T / (R + eps))
        metabolic_cost = clamp_mul(metabolic_cost, float(cfg.metabolic_cost), lo=0.25, hi=50.0)

        # Carriers in a hotter medium (higher damping) should not be given an
        # unrealistically long grace period before dissolution.
        # This uses only existing reference gamma and the emergent gamma.
        starve_window = int(max(1, round(float(cfg.starve_window) * (float(cfg.gamma) / max(gamma, eps)))))

        # ------------------------------------------------------------
        # Genesis
        # ------------------------------------------------------------
        # When resources are scarce, require more unbound pressure to nucleate.
        genesis_pressure = float(cfg.genesis_pressure) / (R + eps)
        genesis_pressure = clamp_mul(genesis_pressure, float(cfg.genesis_pressure), lo=0.25, hi=50.0)

        # Hotter medium => stricter coherence gate (noise suppression).
        genesis_coh = float(cfg.genesis_coherence_threshold) * T
        genesis_coh = float(max(0.0, min(1.0, genesis_coh)))

        # "Unbound" threshold is also resource-dependent:
        # scarce medium -> treat weak bonds as "good enough" binding, suppressing
        # runaway genesis.
        unbound_threshold = float(cfg.unbound_threshold) * R
        unbound_threshold = float(max(p_snap, min(float(cfg.unbound_threshold) * 2.0, unbound_threshold)))

        # ------------------------------------------------------------
        # Mitosis
        # ------------------------------------------------------------
        # Hot/noisy medium -> be more tolerant (lower threshold) to avoid
        # division cascades driven by noise.
        d_threshold = float(cfg.d_threshold) / max(T, eps)
        d_threshold = float(max(0.70, min(1.20, d_threshold)))

        # Scale the persistence window with bond mobility.
        # If bonds evolve slowly (tau_p large), demand longer persistence before splitting.
        mitosis_window = int(max(1, round(float(cfg.mitosis_window) * (tau_p / max(float(cfg.tau_p), eps)))))

        # ------------------------------------------------------------
        # Gate-width adaptation
        # ------------------------------------------------------------
        # Cold + resource-rich -> specialize faster (narrow).
        # Hot + resource-poor  -> explore faster (widen).
        gate_narrow_rate = float(cfg.gate_narrow_rate) * (R / (T + eps))
        gate_widen_rate = float(cfg.gate_widen_rate) * (T / (R + eps))

        gate_narrow_rate = clamp_mul(gate_narrow_rate, float(cfg.gate_narrow_rate), lo=0.05, hi=10.0)
        gate_widen_rate = clamp_mul(gate_widen_rate, float(cfg.gate_widen_rate), lo=0.05, hi=10.0)

        # ------------------------------------------------------------
        # Oscillator amplitude dynamics (Eq. 17)
        # ------------------------------------------------------------
        osc_amp_damping = float(cfg.osc_amp_damping) * T
        osc_amp_damping = clamp_mul(osc_amp_damping, float(cfg.osc_amp_damping), lo=0.25, hi=10.0)

        osc_amp_saturation = float(cfg.osc_amp_saturation) / (R + eps)
        osc_amp_saturation = clamp_mul(osc_amp_saturation, float(cfg.osc_amp_saturation), lo=0.25, hi=50.0)

        return InstantPhysics(
            dt=float(cfg.dt),
            temperature=T,
            resource=R,
            gamma=gamma,
            beta=beta,
            tau_p=tau_p,
            alpha_p=alpha_p,
            lambda_p=lambda_p,
            p_snap=p_snap,
            d_threshold=d_threshold,
            mitosis_window=mitosis_window,
            coherence_tau=coherence_tau,
            intake_tau=intake_tau,
            metabolic_cost=metabolic_cost,
            starve_window=starve_window,
            unbound_threshold=unbound_threshold,
            genesis_pressure=genesis_pressure,
            genesis_coherence_threshold=genesis_coh,
            gate_narrow_rate=gate_narrow_rate,
            gate_widen_rate=gate_widen_rate,
            osc_amp_damping=osc_amp_damping,
            osc_amp_saturation=osc_amp_saturation,
        )


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
        # Each oscillator has: phase, omega (angular freq), amplitude, drive_remaining, drive_strength
        self.phases: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self.omegas: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self.amplitudes: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        self.drive_remaining: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Track the drive strength (signal amplitude) separately
        self.drive_strength: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
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
        self.drive_strength = torch.cat([
            self.drive_strength,
            torch.tensor([signal.amplitude], dtype=DTYPE_REAL, device=self.device)
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
        self.drive_strength = self.drive_strength[keep_mask]
        self._ids = [self._ids[i] for i in range(len(self._ids)) if keep_mask[i]]

    def step_amplitudes(self, dt: float, *, damping: float, saturation: float) -> None:
        """
        Update amplitudes based on drive status AND signal amplitude.

        CHANGE:
        - The previous implementation used fixed defaults damping=1.5 and
          saturation=0.2. Those are now provided by InstantPhysics so they can
          emerge from the medium state (temperature/resource).
        """
        # Decrease drive remaining
        self.drive_remaining = torch.clamp(self.drive_remaining - dt, min=0.0)

        # Drive is signal_amplitude when active, 0 when expired
        active_mask = (self.drive_remaining > 0).float()
        drive = active_mask * self.drive_strength

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
        # Intrinsic frequencies (for rotation) - spectral center
        self.omegas: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Per-carrier gate width
        self.gate_widths: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Coherence EMA (D)
        self.coherence_ema: torch.Tensor = torch.empty(0, dtype=DTYPE_REAL, device=device)
        # Intake EMA for metabolism
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
        """
        return (torch.cos(self.phases) >= 0).float()

    def add(self, phase: float, omega_hz: float, t: float, gate_width: Optional[float] = None) -> int:
        """
        Add a new carrier. Returns its ID.
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
            torch.tensor([1.0], dtype=DTYPE_REAL, device=self.device)
        ])
        self.intake_ema = torch.cat([
            self.intake_ema,
            torch.tensor([0.5], dtype=DTYPE_REAL, device=self.device)
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
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_tuning_strength(
    carrier_phases: torch.Tensor,  # [M]
    osc_phases: torch.Tensor,      # [N]
    gate_width: float
) -> torch.Tensor:
    """Uniform-width tuning strength (kept for backward compatibility)."""
    gate_centers = carrier_phases  # [M]
    osc_peaks = osc_phases         # [N]
    raw_diff = gate_centers.unsqueeze(0) - osc_peaks.unsqueeze(1)
    diff = wrap_to_pi(raw_diff)
    sigma = (gate_width / 2) ** 2
    return torch.exp(-(diff ** 2) / sigma)


def compute_tuning_strength_per_carrier(
    carrier_phases: torch.Tensor,     # [M]
    osc_phases: torch.Tensor,         # [N]
    carrier_gate_widths: torch.Tensor # [M]
) -> torch.Tensor:
    """Per-carrier tuning strength (specialization)."""
    if carrier_phases.numel() == 0 or osc_phases.numel() == 0:
        device = carrier_phases.device if carrier_phases.numel() > 0 else osc_phases.device
        return torch.empty(
            osc_phases.numel(),
            carrier_phases.numel(),
            dtype=DTYPE_REAL,
            device=device,
        )

    raw_diff = carrier_phases.unsqueeze(0) - osc_phases.unsqueeze(1)
    diff = wrap_to_pi(raw_diff)
    sigma = (carrier_gate_widths / 2) ** 2
    return torch.exp(-(diff ** 2) / sigma.unsqueeze(0))


def compute_alignment(carrier_phases: torch.Tensor, osc_phases: torch.Tensor) -> torch.Tensor:
    """DEPRECATED - kept for backward compatibility with tests."""
    phase_diff = carrier_phases.unsqueeze(0) - osc_phases.unsqueeze(1)
    return torch.cos(phase_diff)


def compute_carrier_drive(
    osc_phasors: torch.Tensor,   # [N] complex
    carrier_gates: torch.Tensor, # [M] binary
    tuning: torch.Tensor         # [N, M]
) -> torch.Tensor:
    """
    u_k = G(ψ_k) · Σᵢ T_ik · z_i
    """
    if carrier_gates.numel() == 0:
        return torch.empty(0, dtype=DTYPE_COMPLEX, device=carrier_gates.device)
    if tuning.numel() == 0 or osc_phasors.numel() == 0:
        return torch.zeros(carrier_gates.shape[0], dtype=DTYPE_COMPLEX, device=carrier_gates.device)

    u_raw = torch.matmul(tuning.T.to(DTYPE_COMPLEX), osc_phasors)
    return carrier_gates.to(DTYPE_COMPLEX) * u_raw


def compute_back_influence(
    carrier_c: torch.Tensor,  # [M] complex
    tuning: torch.Tensor,     # [N, M]
    P: Optional[torch.Tensor] = None  # [N, M]
) -> torch.Tensor:
    """Compute back-influence g_i = Σ_k (T_ik * P_ik) c_k (or T_ik c_k if no P)."""
    if tuning.numel() == 0 or carrier_c.numel() == 0:
        N = tuning.shape[0] if tuning.numel() > 0 else 0
        return torch.zeros(N, dtype=DTYPE_COMPLEX, device=tuning.device)

    effective_coupling = tuning * P if (P is not None and P.numel() > 0) else tuning
    return torch.matmul(effective_coupling.to(DTYPE_COMPLEX), carrier_c)


def compute_phase_influence(
    g: torch.Tensor,           # [N] complex
    osc_phases: torch.Tensor,  # [N]
) -> torch.Tensor:
    """Δφ̇_i = Im(g_i · e^(-iφ_i))"""
    if g.numel() == 0:
        return torch.empty(0, dtype=DTYPE_REAL, device=g.device)
    rotated = g * torch.exp(-1j * osc_phases.to(DTYPE_COMPLEX))
    return rotated.imag.to(DTYPE_REAL)


# =============================================================================
# Spectral Profile Computation
# =============================================================================

def compute_spectral_profiles(
    osc_omegas: torch.Tensor,      # [N]
    osc_amplitudes: torch.Tensor,  # [N]
    P: torch.Tensor,               # [N, M]
    tuning: torch.Tensor           # [N, M]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute spectral profile for each carrier.
    """
    if P.numel() == 0 or osc_omegas.numel() == 0:
        device = P.device if P.numel() > 0 else osc_omegas.device
        empty_real = torch.empty(0, dtype=DTYPE_REAL, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        return empty_real, empty_real, empty_bool

    W = P * osc_amplitudes.unsqueeze(1) * tuning
    W_sum = W.sum(dim=0, keepdim=True) + 1e-8
    W_norm = W / W_sum

    omega_center = (W_norm * osc_omegas.unsqueeze(1)).sum(dim=0)

    omega_diff = osc_omegas.unsqueeze(1) - omega_center.unsqueeze(0)
    spectral_variance = (W_norm * (omega_diff ** 2)).sum(dim=0)

    omega_range = osc_omegas.max() - osc_omegas.min() + 1e-8
    threshold = (omega_range / 4) ** 2
    is_multimodal = spectral_variance > threshold

    return omega_center, spectral_variance, is_multimodal


def partition_oscillators_by_frequency(
    osc_omegas: torch.Tensor,        # [N]
    P_col: torch.Tensor,             # [N]
    tuning_col: torch.Tensor,        # [N]
    osc_amplitudes: torch.Tensor,    # [N]
    *,
    weight_threshold: float = 0.01,  # CHANGE: explicit parameter so caller can tie it to p_snap.
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Partition bonded oscillators into two frequency clusters.

    CHANGE:
    - Previously the "significant" cutoff was hard-coded as 0.01.
      We keep 0.01 as the default for backward compatibility, but we make it an
      explicit parameter so the engine can couple it to the (possibly emergent)
      bond snap threshold instead of an orphan constant.
    """
    device = osc_omegas.device

    W = P_col * osc_amplitudes * tuning_col

    significant = W > float(weight_threshold)

    if significant.sum() < 2:
        return significant, torch.zeros_like(significant)

    sig_indices = torch.where(significant)[0]
    sig_omegas = osc_omegas[sig_indices]
    sig_weights = W[sig_indices]

    sorted_indices = torch.argsort(sig_omegas)
    sorted_weights = sig_weights[sorted_indices]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    total = cumsum[-1]

    median_idx = torch.searchsorted(cumsum, total / 2)

    cluster_low = torch.zeros_like(significant)
    cluster_high = torch.zeros_like(significant)

    median_idx_i = int(median_idx.item())
    max_i = int(sorted_indices.numel() - 1)
    median_idx_i = min(median_idx_i, max_i)
    median_omega = sig_omegas[sorted_indices[median_idx_i]]

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
    osc_phasors: torch.Tensor,     # [N] complex
    P: torch.Tensor,               # [N, M]
    osc_amplitudes: torch.Tensor,  # [N]
    tuning: Optional[torch.Tensor] = None  # [N, M]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute coherence statistics for each carrier

    The coherence score is the ratio of the carrier drive to the sum of the weights.
    The baseline is the expected coherence score under random phase.
    The division score is the ratio of the coherence score to the baseline.

    Returns: (coh, baseline, D) tensors of shape [M]
    """
    if P.numel() == 0 or osc_phasors.numel() == 0:
        empty = torch.empty(0, dtype=DTYPE_REAL, device=P.device)
        return empty, empty, empty

    effective = tuning * P if (tuning is not None and tuning.numel() > 0) else P
    W = effective * osc_amplitudes.unsqueeze(1)

    u = torch.matmul(effective.T.to(DTYPE_COMPLEX), osc_phasors)

    sum_w = W.sum(dim=0) + 1e-10
    coh = torch.abs(u) / sum_w

    sum_w_sq = (W ** 2).sum(dim=0)
    baseline = math.sqrt(math.pi / 4) * torch.sqrt(sum_w_sq) / sum_w

    D = coh / (baseline + 1e-10)

    return coh.to(DTYPE_REAL), baseline, D


# =============================================================================
# Engine
# =============================================================================

@dataclass
class EngineEvents:
    births: list[dict] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)
    mitoses: list[dict] = field(default_factory=list)


class ResonantEngine:
    """
    The core resonant compression engine.
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

        # Emergent medium + instantaneous physics snapshot (NEW)
        self.medium = MediumState(device=device)
        self.phys = self.medium.instant(self.config)

        # Time
        self.t = 0.0

        # Events
        self.events = EngineEvents()

        # Top-down feedback buffer (per-carrier, index-aligned)
        self._top_down_carrier_bias = torch.empty(0, dtype=DTYPE_REAL, device=self.device)

    # -------------------------------------------------------------------------
    # Signal IO
    # -------------------------------------------------------------------------

    def add_signal(self, signal: Signal) -> int:
        """Add an input signal to the system. Returns the oscillator ID."""
        oid = self.oscillators.add(signal)

        # Expand P matrix
        if self.carriers.m > 0:
            if self.P.P.numel() == 0:
                self.P.P = torch.zeros(
                    self.oscillators.n,
                    self.carriers.m,
                    dtype=DTYPE_REAL,
                    device=self.device,
                )
            else:
                self.P.add_oscillator()

        return oid

    def _estimate_input_energy(self, signals: list[Signal]) -> torch.Tensor:
        """Estimate energy injected by input signals (frequency-weighted amplitude)."""
        if not signals:
            return torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
        energy = 0.0
        for sig in signals:
            energy += sig.amplitude * sig.freq_hz
        return torch.tensor(energy, dtype=DTYPE_REAL, device=self.device)

    def _seed_input_bonds(self, osc_indices: list[int]) -> None:
        """Seed bonds from new input oscillators to aligned carriers."""
        if not osc_indices:
            return
        if self.carriers.m == 0 or self.P.P.numel() == 0:
            return

        cfg = self.config
        osc_phases = self.oscillators.phases[osc_indices]
        tuning = compute_tuning_strength_per_carrier(
            self.carriers.phases,
            osc_phases,
            self.carriers.gate_widths,
        )

        max_bonds = max(1, int(cfg.input_max_bonds_per_signal))
        threshold = float(cfg.input_bond_threshold)
        strength = float(cfg.input_bond_strength)

        for row_idx, osc_idx in enumerate(osc_indices):
            row = tuning[row_idx]
            if row.numel() == 0:
                continue
            valid = row >= threshold
            if not torch.any(valid):
                continue
            scores = row.clone()
            scores[~valid] = 0.0
            k = min(max_bonds, scores.numel())
            top_vals, top_idx = torch.topk(scores, k=k)
            for val, car_idx in zip(top_vals, top_idx):
                if float(val) <= 0.0:
                    continue
                self.P.seed_bond(osc_idx, int(car_idx), strength)

    def _seed_bonds_to_first_carrier(self, osc_indices: list[int]) -> None:
        """Force bonds from new oscillators to carrier 0."""
        if not osc_indices:
            return
        if self.carriers.m == 0:
            return
        if self.P.P.numel() == 0:
            self.P.P = torch.zeros(
                self.oscillators.n,
                self.carriers.m,
                dtype=DTYPE_REAL,
                device=self.device,
            )
        strength = float(self.config.input_bond_strength)
        for osc_idx in osc_indices:
            self.P.seed_bond(osc_idx, 0, strength)

    # -------------------------------------------------------------------------
    # Genesis / birth
    # -------------------------------------------------------------------------

    def _nucleate_carrier(self, from_osc_idx: int) -> int:
        """
        Nucleate a new carrier from an unbound oscillator.

        CHANGE:
        - Birth log now includes medium state for auditability of emergent physics.
        """
        phase = float(self.oscillators.phases[from_osc_idx])
        omega = float(self.oscillators.omegas[from_osc_idx])
        freq_hz = omega / (2 * math.pi)

        cid = self.carriers.add(phase, freq_hz, self.t)

        # Keep top-down buffer aligned with carriers
        self._top_down_carrier_bias = torch.cat(
            [self._top_down_carrier_bias, torch.zeros(1, dtype=DTYPE_REAL, device=self.device)]
        )

        # Expand P matrix
        if self.oscillators.n > 0:
            if self.P.P.numel() == 0:
                self.P.P = torch.zeros(
                    self.oscillators.n, 1,
                    dtype=DTYPE_REAL, device=self.device
                )
            else:
                self.P.add_carrier()

        # Seed initial bond with triggering oscillator
        seed_strength = 0.5
        self.P.seed_bond(from_osc_idx, self.carriers.m - 1, seed_strength)

        self.events.births.append(
            {
                "t": self.t,
                "from_osc_idx": from_osc_idx,
                "carrier_id": cid,
                "phase": phase,
                "omega": omega,
                "omega_hz": freq_hz,
                "gate_width_init": float(self.carriers.gate_widths[-1]),
                "seed_strength": seed_strength,
                # NEW: medium snapshot at birth
                "medium_T": float(self.phys.temperature),
                "medium_R": float(self.phys.resource),
            }
        )

        return cid

    def _check_genesis(self) -> None:
        """
        Check for genesis condition: unbound oscillators with coherent high energy.

        CHANGE:
        - Uses emergent thresholds from InstantPhysics instead of static config
          (unbound_threshold, genesis_pressure, genesis_coherence_threshold).
        """
        if self.oscillators.n == 0:
            return
        if self.oscillators.amplitudes.numel() == 0:
            return

        cfg = self.config
        phys = self.phys

        if self.carriers.m == 0:
            # Genesis coherence gating applies even when no carriers exist.
            unbound_amp = self.oscillators.amplitudes
            pressure = unbound_amp.sum()
            if pressure <= phys.genesis_pressure:
                return

            unbound_phases = self.oscillators.phases
            if unbound_phases.numel() >= 2:
                phasors = torch.exp(1j * unbound_phases.to(DTYPE_COMPLEX))
                weights = unbound_amp / (unbound_amp.sum() + 1e-8)
                weighted_phasor = (weights.to(DTYPE_COMPLEX) * phasors).sum()
                R_unbound = float(torch.abs(weighted_phasor))
                if R_unbound < phys.genesis_coherence_threshold:
                    return

            strongest_idx = int(unbound_amp.argmax())
            if strongest_idx < self.oscillators.n:
                self._nucleate_carrier(strongest_idx)
            return

        # Find unbound oscillators
        if self.P.P.numel() > 0 and self.P.P.shape[0] == self.oscillators.n:
            max_bonds = self.P.P.max(dim=1).values
        else:
            max_bonds = torch.zeros(self.oscillators.n, device=self.device)

        unbound_mask = max_bonds < phys.unbound_threshold
        if not unbound_mask.any():
            return

        unbound_amp = self.oscillators.amplitudes[unbound_mask]
        pressure = unbound_amp.sum()
        if pressure <= phys.genesis_pressure:
            return

        unbound_phases = self.oscillators.phases[unbound_mask]
        if unbound_phases.numel() >= 2:
            phasors = torch.exp(1j * unbound_phases.to(DTYPE_COMPLEX))
            weights = unbound_amp / (unbound_amp.sum() + 1e-8)
            weighted_phasor = (weights.to(DTYPE_COMPLEX) * phasors).sum()
            R_unbound = float(torch.abs(weighted_phasor))
            if R_unbound < phys.genesis_coherence_threshold:
                return

        unbound_indices = torch.where(unbound_mask)[0]
        strongest_idx = unbound_indices[unbound_amp.argmax()]
        if strongest_idx < self.oscillators.n:
            self._nucleate_carrier(int(strongest_idx))

    # -------------------------------------------------------------------------
    # Mitosis
    # -------------------------------------------------------------------------

    def _check_mitosis(self) -> None:
        """
        Check for mitosis condition: carriers with persistent spectral conflict.

        CHANGE:
        - Uses emergent mitosis threshold/window/coherence_tau from InstantPhysics.
        """
        if self.carriers.m == 0 or self.oscillators.n < 2:
            return

        cfg = self.config
        phys = self.phys

        tuning = compute_tuning_strength_per_carrier(
            self.carriers.phases,
            self.oscillators.phases,
            self.carriers.gate_widths,
        )

        _, _, D = compute_coherence(
            self.oscillators.phasors,
            self.P.P,
            self.oscillators.amplitudes,
            tuning=tuning,
        )

        _, spectral_var, is_multimodal = compute_spectral_profiles(
            self.oscillators.omegas,
            self.oscillators.amplitudes,
            self.P.P,
            tuning
        )

        # Update coherence EMA using emergent coherence_tau
        if D.numel() > 0:
            alpha = cfg.dt / max(phys.coherence_tau, cfg.dt)
            self.carriers.coherence_ema = (1 - alpha) * self.carriers.coherence_ema + alpha * D

        below = (D < phys.d_threshold) & is_multimodal
        self.carriers.d_below_threshold = torch.where(
            below,
            self.carriers.d_below_threshold + 1,
            torch.zeros_like(self.carriers.d_below_threshold)
        )

        mitosis_mask = self.carriers.d_below_threshold >= int(phys.mitosis_window)

        for idx in torch.where(mitosis_mask)[0].tolist():
            d_parent = float(D[idx].detach().cpu()) if D.numel() > 0 else float("nan")
            spectral_var_parent = float(spectral_var[idx].detach().cpu()) if spectral_var.numel() > 0 else float("nan")
            multimodal_flag = bool(is_multimodal[idx].detach().cpu()) if is_multimodal.numel() > 0 else False
            self._do_spectral_mitosis(
                idx,
                tuning,
                d_parent=d_parent,
                spectral_var_parent=spectral_var_parent,
                multimodal_flag=multimodal_flag,
            )
            self.carriers.d_below_threshold[idx] = 0

    def _do_spectral_mitosis(
        self,
        carrier_idx: int,
        tuning: torch.Tensor,
        *,
        d_parent: float,
        spectral_var_parent: float,
        multimodal_flag: bool,
    ) -> None:
        """Perform spectral factorization."""
        parent_id = self.carriers._ids[carrier_idx]
        cfg = self.config
        phys = self.phys

        P_col = self.P.P[:, carrier_idx]
        tuning_col = tuning[:, carrier_idx]

        cluster_low, cluster_high = partition_oscillators_by_frequency(
            self.oscillators.omegas,
            P_col,
            tuning_col,
            self.oscillators.amplitudes,
            weight_threshold=phys.p_snap,  # CHANGE: tie to snap threshold
        )

        if cluster_low.sum() == 0 or cluster_high.sum() == 0:
            return

        W = P_col * self.oscillators.amplitudes * tuning_col

        W_low = W * cluster_low.float()
        omega_low = (W_low * self.oscillators.omegas).sum() / (W_low.sum() + 1e-12)

        W_high = W * cluster_high.float()
        omega_high = (W_high * self.oscillators.omegas).sum() / (W_high.sum() + 1e-12)

        self.carriers.omegas[carrier_idx] = omega_low
        self.P.P[cluster_high, carrier_idx] = 0.0

        parent_c = self.carriers.c[carrier_idx]
        phase = float(torch.angle(parent_c))
        freq_hz_high = float(omega_high) / (2 * math.pi)
        parent_gate_width = float(self.carriers.gate_widths[carrier_idx])

        cid = self.carriers.add(phase, freq_hz_high, self.t, gate_width=parent_gate_width)

        self._top_down_carrier_bias = torch.cat(
            [self._top_down_carrier_bias, torch.zeros(1, dtype=DTYPE_REAL, device=self.device)]
        )

        if self.P.P.numel() > 0:
            self.P.add_carrier()
            self.P.P[:, -1] = P_col * cluster_high.float()

        # Small symmetry-breaking noise.
        # CHANGE: scale by medium temperature (hotter medium -> noisier splits).
        noise_std = 0.05 * float(phys.temperature)
        noise_parent = noise_std * (torch.randn(2, generator=self.rng).to(self.device))
        noise_child = noise_std * (torch.randn(2, generator=self.rng).to(self.device))

        self.carriers.c[carrier_idx] += (noise_parent[0] + 1j * noise_parent[1])
        self.carriers.c[-1] += (noise_child[0] + 1j * noise_child[1])

        eps = 1e-12
        low_total = float((P_col * cluster_low.float()).sum().detach().cpu()) + eps
        high_total = float((P_col * cluster_high.float()).sum().detach().cpu()) + eps
        purity_low = float((P_col * cluster_low.float()).sum().detach().cpu()) / low_total
        purity_high = float((P_col * cluster_high.float()).sum().detach().cpu()) / high_total

        omega_low_hz = float(omega_low) / (2 * math.pi)
        omega_high_hz = freq_hz_high

        self.events.mitoses.append(
            {
                "t": self.t,
                "parent_id": parent_id,
                "child_id": cid,
                "D_parent": d_parent,
                "spectral_var_parent": spectral_var_parent,
                "multimodal_flag": multimodal_flag,
                "purity_low": purity_low,
                "purity_high": purity_high,
                "delta_centers_hz": abs(omega_high_hz - omega_low_hz),
                "parent_idx": carrier_idx,
                "parent_omega_hz": omega_low_hz,
                "child_omega_hz": omega_high_hz,
                "low_cluster_size": int(cluster_low.sum()),
                "high_cluster_size": int(cluster_high.sum()),
                # NEW: medium snapshot
                "medium_T": float(phys.temperature),
                "medium_R": float(phys.resource),
            }
        )

    # -------------------------------------------------------------------------
    # Dissolution / death
    # -------------------------------------------------------------------------

    def _check_dissolution(self) -> None:
        """
        Dissolution: carriers that fail to capture sufficient coherent energy.

        CHANGE:
        - Uses emergent metabolic_cost and starve_window from InstantPhysics.
        """
        if self.carriers.m == 0:
            return

        cfg = self.config
        phys = self.phys

        effective_intake = self.carriers.intake_ema * self.carriers.coherence_ema

        starving = effective_intake < phys.metabolic_cost
        if int(cfg.carrier_grace_steps) > 0:
            ages = (self.t - self.carriers.birth_t) / float(cfg.dt)
            grace_mask = ages < float(cfg.carrier_grace_steps)
            if grace_mask.any():
                starving = torch.where(grace_mask, torch.zeros_like(starving), starving)
        self.carriers.starve_steps = torch.where(
            starving,
            self.carriers.starve_steps + 1,
            torch.zeros_like(self.carriers.starve_steps)
        )

        dissolve_mask = self.carriers.starve_steps >= int(phys.starve_window)

        if dissolve_mask.any():
            old_m = int(self.carriers.m)
            indices = torch.where(dissolve_mask)[0]
            death_events = self.carriers.remove(indices)
            self.P.remove_carriers(indices)

            if self._top_down_carrier_bias.numel() == old_m:
                keep_mask = torch.ones(old_m, dtype=torch.bool, device=self.device)
                keep_mask[indices] = False
                self._top_down_carrier_bias = self._top_down_carrier_bias[keep_mask]
            else:
                self._top_down_carrier_bias = torch.zeros(self.carriers.m, dtype=DTYPE_REAL, device=self.device)

            for event in death_events:
                event["t"] = self.t
                event["reason"] = "starvation"
                event["carrier_id"] = event.get("id")
                birth_t = float(event.get("birth_t", 0.0))
                age = float(self.t - birth_t)
                event["age"] = age
                final_coherence = float(event.get("final_coherence", 0.0))
                intake_raw = float(event.get("intake", 0.0))
                event["intake_eff"] = intake_raw * final_coherence
                # NEW: medium snapshot
                event["medium_T"] = float(phys.temperature)
                event["medium_R"] = float(phys.resource)
                self.events.deaths.append(event)

    # -------------------------------------------------------------------------
    # Gate width adaptation
    # -------------------------------------------------------------------------

    def _update_gate_widths(self) -> None:
        """
        Update gate widths based on coherence history.

        CHANGE:
        - Uses emergent gate_narrow_rate / gate_widen_rate from InstantPhysics.
        """
        if self.carriers.m == 0:
            return

        cfg = self.config
        phys = self.phys
        dt = cfg.dt

        D = self.carriers.coherence_ema

        dW_narrow = -phys.gate_narrow_rate * (D - 1.0) * self.carriers.gate_widths
        dW_widen = phys.gate_widen_rate * (1.0 - D) * self.carriers.gate_widths

        dW = torch.where(D > 1.0, dW_narrow, dW_widen)

        # Top-down shaping (unchanged logic, but now combined with emergent base rates)
        if (
            cfg.top_down_gate_rate != 0.0
            and self._top_down_carrier_bias.numel() == self.carriers.m
            and self.carriers.m > 0
        ):
            support = self.carriers.intake_ema
            support = support / (support.max() + 1e-8)
            support = torch.clamp(support, 0.0, 1.0)

            coherence_excess = torch.relu(D - 1.0)
            coherent_noise = coherence_excess * (1.0 - support)

            bias = torch.clamp(self._top_down_carrier_bias, -cfg.top_down_max_abs, cfg.top_down_max_abs)
            utility = bias * support - cfg.top_down_noise_penalty * coherent_noise

            dW = dW + (-cfg.top_down_gate_rate * utility * self.carriers.gate_widths)

        self.carriers.gate_widths = torch.clamp(
            self.carriers.gate_widths + dW * dt,
            min=cfg.gate_width_min,
            max=cfg.gate_width_max
        )

    # -------------------------------------------------------------------------
    # Bonds
    # -------------------------------------------------------------------------

    def _update_bonds(self) -> None:
        """
        Update elastic bonds P based on resonant capture.

        CHANGE (important):
        - Bond rates now use emergent (InstantPhysics) tau_p/alpha_p/lambda_p.
        - Tuning uses PER-CARRIER gate widths (specialization) instead of the
          old uniform cfg.gate_width, so bonds and drive are consistent.
        - The growth law is simplified to remove a hard-coded formation factor (0.1).
          We use a logistic-style (1 - P) factor so bonds can form from zero and
          saturate naturally, without an extra arbitrary timescale.
        - Capture is scaled by oscillator amplitude so weak oscillators do not
          create bonds simply because phases line up.
        """
        if self.oscillators.n == 0 or self.carriers.m == 0:
            return

        if self.P.P.shape != (self.oscillators.n, self.carriers.m):
            self.P.resize(self.oscillators.n, self.carriers.m)

        cfg = self.config
        phys = self.phys
        dt = cfg.dt

        gates = self.carriers.gate()

        # CHANGE: per-carrier tuning (matches the drive computation)
        tuning = compute_tuning_strength_per_carrier(
            self.carriers.phases,
            self.oscillators.phases,
            self.carriers.gate_widths,
        )

        # Energy capture indicator (projection) scaled by oscillator amplitude
        c_expanded = self.carriers.c.unsqueeze(0)           # [1, M]
        phi_expanded = self.oscillators.phases.unsqueeze(1) # [N, 1]
        capture = torch.real(c_expanded * torch.exp(-1j * phi_expanded.to(DTYPE_COMPLEX)))  # [N, M]
        capture = capture * self.oscillators.amplitudes.unsqueeze(1)  # [N, M]

        capture_pos = torch.clamp(capture, min=0.0)

        # Logistic growth: bonds form from 0 and saturate at 1 without extra constants.
        growth = (
            phys.alpha_p *
            gates.unsqueeze(0) *
            tuning *
            capture_pos *
            (1.0 - self.P.P)
        )

        decay = phys.lambda_p * self.P.P

        dP = (growth - decay) / max(phys.tau_p, 1e-6)
        self.P.P = torch.clamp(self.P.P + dP * dt, min=0.0, max=1.0)

        self.P.P = torch.where(
            self.P.P < phys.p_snap,
            torch.zeros_like(self.P.P),
            self.P.P
        )

    # -------------------------------------------------------------------------
    # Carrier dynamics
    # -------------------------------------------------------------------------

    def _update_carriers(self, drive: torch.Tensor) -> None:
        """
        dc/dt = iωc + u - γc - β|c|²c

        CHANGE:
        - Uses emergent gamma/beta/intake_tau from InstantPhysics.
        """
        if self.carriers.m == 0:
            return

        cfg = self.config
        phys = self.phys
        dt = cfg.dt

        rotation = 1j * self.carriers.omegas.to(DTYPE_COMPLEX) * self.carriers.c

        damping = phys.gamma * self.carriers.c
        saturation = phys.beta * (torch.abs(self.carriers.c) ** 2) * self.carriers.c

        dc = rotation + drive - damping - saturation
        self.carriers.c = self.carriers.c + dc * dt

        intake = torch.abs(drive)
        alpha_ema = dt / max(phys.intake_tau, dt)
        self.carriers.intake_ema = (1 - alpha_ema) * self.carriers.intake_ema + alpha_ema * intake

    # -------------------------------------------------------------------------
    # Oscillator dynamics
    # -------------------------------------------------------------------------

    def _update_oscillators(self, phase_influence: torch.Tensor) -> None:
        """
        dφ/dt = ω + phase_influence

        CHANGE:
        - Oscillator amplitude damping/saturation now come from InstantPhysics.
        """
        if self.oscillators.n == 0:
            return

        cfg = self.config
        phys = self.phys
        dt = cfg.dt

        dphase = self.oscillators.omegas + phase_influence
        self.oscillators.phases = (self.oscillators.phases + dphase * dt) % (2 * math.pi)

        self.oscillators.step_amplitudes(
            dt,
            damping=phys.osc_amp_damping,
            saturation=phys.osc_amp_saturation,
        )

    # -------------------------------------------------------------------------
    # Housekeeping
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def step(self, signals: Optional[list[Signal]] = None) -> None:
        """
        Advance the simulation by one time step.

        CHANGE:
        - At the start of the step, derive InstantPhysics from MediumState.
        - At the end of the step, update MediumState from the resulting dynamics.
        """
        cfg = self.config

        # (0) Refresh instantaneous physics from medium state (NEW)
        self.phys = self.medium.instant(cfg)
        phys = self.phys

        # (1) Add new signals
        input_energy = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
        new_osc_indices: list[int] = []
        if signals:
            input_energy = self._estimate_input_energy(signals)
            for sig in signals:
                osc_idx = self.oscillators.n
                self.add_signal(sig)
                new_osc_indices.append(osc_idx)

        # (1a) Genesis rule: first signal creates the first carrier (no other genesis)
        if cfg.genesis_from_first_signal_only:
            if self.carriers.m == 0:
                if new_osc_indices:
                    self._nucleate_carrier(new_osc_indices[0])
                elif self.oscillators.n > 0:
                    self._nucleate_carrier(0)
            # All additional signals bond to the first carrier until split.
            if new_osc_indices and self.carriers.m > 0:
                self._seed_bonds_to_first_carrier(new_osc_indices)
        else:
            # (1a) Optional: force genesis from first input (useful for audio demos)
            if cfg.force_genesis_on_input and self.carriers.m == 0 and new_osc_indices:
                self._nucleate_carrier(new_osc_indices[0])

            # (1b) Seed bonds from aligned carriers (input -> oscillator coupling)
            if new_osc_indices:
                self._seed_input_bonds(new_osc_indices)

        # (2) Genesis
        if not cfg.genesis_from_first_signal_only:
            self._check_genesis()

        # (2b) Decay stored top-down feedback
        if self._top_down_carrier_bias.numel() == self.carriers.m and self.carriers.m > 0:
            if cfg.top_down_tau > 0:
                decay = math.exp(-cfg.dt / cfg.top_down_tau)
                self._top_down_carrier_bias = self._top_down_carrier_bias * decay
            else:
                self._top_down_carrier_bias = torch.zeros(self.carriers.m, dtype=DTYPE_REAL, device=self.device)

        # (3) Compute tuning + drive
        if self.carriers.m > 0 and self.oscillators.n > 0:
            tuning = compute_tuning_strength_per_carrier(
                self.carriers.phases,
                self.oscillators.phases,
                self.carriers.gate_widths
            )
            gates = self.carriers.gate()
            drive = compute_carrier_drive(
                self.oscillators.phasors,
                gates,
                tuning
            )
        else:
            tuning = torch.empty(
                self.oscillators.n,
                self.carriers.m,
                dtype=DTYPE_REAL,
                device=self.device,
            )
            drive = torch.zeros(self.carriers.m, dtype=DTYPE_COMPLEX, device=self.device)

        # (4) Update carriers
        self._update_carriers(drive)

        # (5) Back-influence
        if self.carriers.m > 0 and self.oscillators.n > 0:
            g = compute_back_influence(self.carriers.c, tuning, self.P.P if self.P.P.numel() > 0 else None)
            phase_influence = compute_phase_influence(g, self.oscillators.phases)
        else:
            phase_influence = torch.zeros(self.oscillators.n, dtype=DTYPE_REAL, device=self.device)

        # (6) Update oscillators
        self._update_oscillators(phase_influence)

        # (7) Update bonds
        self._update_bonds()

        # (8) Mitosis
        self._check_mitosis()

        # (9) Gate widths
        self._update_gate_widths()

        # (10) Dissolution
        self._check_dissolution()

        # (11) Garbage collect
        self._garbage_collect()

        # (12) Update medium state from observables (NEW)
        # Supply: total oscillator amplitude + input energy (free energy proxy)
        if self.oscillators.n > 0:
            supply = self.oscillators.amplitudes.sum()
        else:
            supply = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
        if input_energy.numel() > 0:
            supply = supply + (input_energy * float(cfg.input_energy_scale))

        # Demand: total intake (consumption proxy)
        if self.carriers.m > 0:
            demand = self.carriers.intake_ema.sum()
            # Coherent share: map D in [0,2] -> quality in [0,1]
            quality = torch.clamp(self.carriers.coherence_ema, 0.0, 2.0) / 2.0
            coherent_demand = (self.carriers.intake_ema * quality).sum()
            carrier_energy = (torch.abs(self.carriers.c) ** 2).sum()
        else:
            demand = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
            coherent_demand = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
            carrier_energy = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)

        self.medium.update(
            cfg,
            supply=supply,
            demand=demand,
            coherent_demand=coherent_demand,
            carrier_energy=carrier_energy,
        )

        # Advance time
        self.t += cfg.dt

    # =========================================================================
    # Metrics
    # =========================================================================

    def nnz_P(self) -> int:
        return self.P.nnz()

    def global_sync_R(self) -> float:
        if self.oscillators.n == 0:
            return 0.0

        active = self.oscillators.amplitudes > self.config.noise_floor
        if not active.any():
            return 0.0

        phasors = torch.exp(1j * self.oscillators.phases[active].to(DTYPE_COMPLEX))
        R = torch.abs(phasors.mean())
        return float(R)

    def L_comp(self) -> int:
        return self.nnz_P() + self.carriers.m

    # =========================================================================
    # Observation Interface
    # =========================================================================

    def observe(self, feedback: Optional[dict] = None) -> dict:
        """
        Canonical observation interface.

        CHANGE:
        - Adds medium state + instantaneous physics snapshot to the observation.
          This is crucial when constants are emergent; you need to be able to
          *observe* the values the medium is producing.
        """
        self.provide_feedback(feedback)

        if self.carriers.m > 0 and self.oscillators.n > 0:
            tuning = compute_tuning_strength_per_carrier(
                self.carriers.phases,
                self.oscillators.phases,
                self.carriers.gate_widths
            )

            omega_center, spectral_var, is_multimodal = compute_spectral_profiles(
                self.oscillators.omegas,
                self.oscillators.amplitudes,
                self.P.P,
                tuning
            )

            raw_assignment = self.P.P * tuning
            assignment_sum = raw_assignment.sum(dim=1, keepdim=True) + 1e-8
            soft_assignment = raw_assignment / assignment_sum
        else:
            tuning = torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device)
            omega_center = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            spectral_var = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            is_multimodal = torch.empty(0, dtype=torch.bool, device=self.device)
            soft_assignment = torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device)

        # Keep physics snapshot current for observers
        self.phys = self.medium.instant(self.config)

        return {
            "t": self.t,
            "n_oscillators": self.oscillators.n,
            "n_carriers": self.carriers.m,
            "n_bonds": self.nnz_P(),
            "global_sync_R": self.global_sync_R(),
            "L_comp": self.L_comp(),

            # NEW: medium + emergent physics snapshot
            "medium_temperature": float(self.phys.temperature),
            "medium_resource": float(self.phys.resource),
            "physics": {
                "gamma": float(self.phys.gamma),
                "beta": float(self.phys.beta),
                "tau_p": float(self.phys.tau_p),
                "alpha_p": float(self.phys.alpha_p),
                "lambda_p": float(self.phys.lambda_p),
                "d_threshold": float(self.phys.d_threshold),
                "mitosis_window": int(self.phys.mitosis_window),
                "intake_tau": float(self.phys.intake_tau),
                "coherence_tau": float(self.phys.coherence_tau),
                "metabolic_cost": float(self.phys.metabolic_cost),
                "starve_window": int(self.phys.starve_window),
                "genesis_pressure": float(self.phys.genesis_pressure),
                "genesis_coherence_threshold": float(self.phys.genesis_coherence_threshold),
                "unbound_threshold": float(self.phys.unbound_threshold),
                "gate_narrow_rate": float(self.phys.gate_narrow_rate),
                "gate_widen_rate": float(self.phys.gate_widen_rate),
                "osc_amp_damping": float(self.phys.osc_amp_damping),
                "osc_amp_saturation": float(self.phys.osc_amp_saturation),
            },

            # Carrier-level observables [M]
            "carrier_ids": list(self.carriers._ids),
            "carrier_names": list(self.carriers._names),
            "carrier_energy": self.carriers.energies.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_phase": self.carriers.phases.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_omega": self.carriers.omegas.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_intake": self.carriers.intake_ema.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_coherence": self.carriers.coherence_ema.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_gate_width": self.carriers.gate_widths.clone() if self.carriers.m > 0 else torch.empty(0, device=self.device),
            "carrier_spectral_center": omega_center,
            "carrier_spectral_variance": spectral_var,
            "carrier_is_multimodal": is_multimodal,

            "top_down_carrier_bias": self._top_down_carrier_bias.clone()
            if self._top_down_carrier_bias.numel() == self.carriers.m
            else torch.zeros(self.carriers.m, dtype=DTYPE_REAL, device=self.device),

            # Oscillator-level observables [N]
            "osc_amplitude": self.oscillators.amplitudes.clone() if self.oscillators.n > 0 else torch.empty(0, device=self.device),
            "osc_phase": self.oscillators.phases.clone() if self.oscillators.n > 0 else torch.empty(0, device=self.device),
            "osc_omega": self.oscillators.omegas.clone() if self.oscillators.n > 0 else torch.empty(0, device=self.device),

            # Soft assignment matrix [N, M]
            "soft_assignment": soft_assignment,

            # Tuning matrix [N, M]
            "tuning": tuning,

            # Presence matrix [N, M]
            "presence": self.P.P.clone() if self.P.P.numel() > 0 else torch.empty(0, 0, device=self.device),
        }

    def provide_feedback(self, feedback: Optional[dict]) -> None:
        """Store bounded top-down feedback for later dynamics updates."""
        if feedback is None or self.carriers.m == 0:
            return

        cfg = self.config
        m = int(self.carriers.m)

        carrier_utility = feedback.get("carrier_utility", None)
        if carrier_utility is None:
            return

        u = torch.zeros(m, dtype=DTYPE_REAL, device=self.device)
        if isinstance(carrier_utility, dict):
            id_to_idx = {cid: i for i, cid in enumerate(self.carriers._ids)}
            for k, v in carrier_utility.items():
                key = k
                if key not in id_to_idx:
                    try:
                        key = int(key)  # type: ignore[arg-type]
                    except Exception:
                        key = k
                if key in id_to_idx:
                    u[id_to_idx[key]] = float(v)
        elif isinstance(carrier_utility, torch.Tensor):
            u = carrier_utility.to(device=self.device, dtype=DTYPE_REAL).flatten()
            if u.numel() != m:
                return
        else:
            try:
                if len(carrier_utility) != m:  # type: ignore[arg-type]
                    return
                u = torch.tensor(list(carrier_utility), dtype=DTYPE_REAL, device=self.device)  # type: ignore[arg-type]
            except Exception:
                return

        g = feedback.get("utility", 1.0)
        try:
            g = float(g)
        except Exception:
            g = 1.0
        u = u * g

        u = torch.clamp(u, -cfg.top_down_max_abs, cfg.top_down_max_abs)
        if self._top_down_carrier_bias.numel() != m:
            self._top_down_carrier_bias = torch.zeros(m, dtype=DTYPE_REAL, device=self.device)

        if cfg.top_down_tau > 0:
            alpha = float(cfg.dt / cfg.top_down_tau)
            alpha = max(0.0, min(1.0, alpha))
            self._top_down_carrier_bias = (1.0 - alpha) * self._top_down_carrier_bias + alpha * u
        else:
            self._top_down_carrier_bias = u

    # Compact observation helpers (unchanged except they now inherit emergent physics via observe())

    def observe_carriers_only(self) -> torch.Tensor:
        if self.carriers.m == 0:
            return torch.empty(0, 9, dtype=DTYPE_REAL, device=self.device)

        obs = self.observe()

        features = torch.stack([
            obs["carrier_energy"],
            torch.sin(obs["carrier_phase"]),
            torch.cos(obs["carrier_phase"]),
            obs["carrier_omega"] / (2 * math.pi),
            obs["carrier_intake"],
            obs["carrier_coherence"],
            obs["carrier_gate_width"] / math.pi,
            obs["carrier_spectral_center"] / (2 * math.pi),
            torch.sqrt(obs["carrier_spectral_variance"] + 1e-8) / (2 * math.pi),
        ], dim=1)

        return features

    def observe_global(self) -> torch.Tensor:
        obs = self.observe()

        if obs["n_carriers"] > 0:
            energies = obs["carrier_energy"]
            mean_energy = energies.mean()
            mean_coherence = obs["carrier_coherence"].mean()

            energy_probs = energies / (energies.sum() + 1e-8)
            energy_entropy = -(energy_probs * torch.log(energy_probs + 1e-8)).sum()

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
        lam = self.event_rate_hz * dt

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


class AudioTokenizer:
    """Tokenize audio into Signal frames for cocktail party experiments."""

    def __init__(
        self,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        top_k: int = 6,
        min_energy: float = 0.08,
        amplitude_scale: float = 1.0,
    ):
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.top_k = top_k
        self.min_energy = min_energy
        self.amplitude_scale = amplitude_scale

    @staticmethod
    def _mixdown(audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            return audio.mean(dim=0)
        return audio

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    def _stft(self, audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        frame_len = max(16, int(round(self.frame_ms * sample_rate / 1000.0)))
        hop_len = max(8, int(round(self.hop_ms * sample_rate / 1000.0)))
        n_fft = self._next_pow2(frame_len)
        window = torch.hann_window(n_fft, device=audio.device)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_len,
            window=window,
            return_complex=True,
        )
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(audio.device)
        return spec, freqs, frame_len, hop_len

    def tokenize(self, audio_stream: torch.Tensor, sample_rate: int) -> list[list[Signal]]:
        """Convert audio into per-frame Signal lists."""
        audio = self._mixdown(audio_stream).to(dtype=DTYPE_REAL)
        if audio.numel() == 0:
            return []
        peak = float(audio.abs().max())
        if peak > 0:
            audio = audio / peak

        spec, freqs, frame_len, hop_len = self._stft(audio, sample_rate)
        n_frames = spec.shape[1]
        signals_by_frame: list[list[Signal]] = []
        frame_dt = frame_len / float(sample_rate)

        for t in range(n_frames):
            frame_spec = spec[:, t]
            mags = frame_spec.abs()
            if mags.numel() == 0:
                signals_by_frame.append([])
                continue

            mags = mags.clone()
            mags[0] = 0.0  # drop DC
            frame_max = float(mags.max())
            if frame_max <= 0:
                signals_by_frame.append([])
                continue

            threshold = frame_max * self.min_energy
            valid = mags >= threshold
            if not torch.any(valid):
                signals_by_frame.append([])
                continue

            mags_valid = mags.clone()
            mags_valid[~valid] = 0.0
            k = min(self.top_k, mags_valid.numel())
            top_vals, top_idx = torch.topk(mags_valid, k=k)

            frame_signals: list[Signal] = []
            for val, idx in zip(top_vals, top_idx):
                amp = float(val) / frame_max
                if amp <= 0:
                    continue
                freq_hz = float(freqs[int(idx)])
                phase = float(torch.angle(frame_spec[int(idx)]))
                frame_signals.append(
                    Signal(
                        freq_hz=freq_hz,
                        phase=phase,
                        amplitude=amp * self.amplitude_scale,
                        duration_s=frame_dt,
                    )
                )

            signals_by_frame.append(frame_signals)

        return signals_by_frame


class CocktailParty:
    """CocktailParty converts a multi-speaker audio stream into signals for the resonant engine.

    It is currently expected that the resonant engine can zero-shot the solution.
    The solution would be for the resonant engine to be able to separate the multiple speakers
    into separate signals, which can then be consumed from the outputs.
    """
    
    def __init__(self, engine: ResonantEngine, tokenizer: Optional[AudioTokenizer] = None):
        self.engine = engine
        self.tokenizer = tokenizer or AudioTokenizer()

    def load_wav(self, path: str | Path) -> tuple[torch.Tensor, int]:
        """Load a wav file and return (audio, sample_rate)."""
        audio, sr = torchaudio.load(str(path))
        return audio, int(sr)

    def tokenize(self, audio_stream: torch.Tensor, sample_rate: int) -> list[list[Signal]]:
        """Tokenize audio into per-frame Signal lists."""
        return self.tokenizer.tokenize(audio_stream, sample_rate)

    def run(self, audio_stream: torch.Tensor, sample_rate: int) -> None:
        """Convert a multi-speaker audio stream into signals and step the engine.

        Signals should be shaped in such a way that they can become the top-layer of the
        tensordict that models the physical state space.
        """
        frames = self.tokenizer.tokenize(audio_stream, sample_rate)
        if not frames:
            return

        frame_dt = self.tokenizer.hop_ms / 1000.0
        steps_per_frame = max(1, int(round(frame_dt / self.engine.config.dt)))

        for frame_signals in frames:
            self.engine.step(frame_signals)
            for _ in range(steps_per_frame - 1):
                self.engine.step()


# =============================================================================
# Main (for quick testing)
# =============================================================================

def main():
    print("=" * 60)
    print("Resonant Compression Systems — Engine Test (Emergent Physics)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    engine = ResonantEngine(seed=0)
    stream = StochasticStream(seed=0)

    dt = engine.config.dt
    steps = int(10.0 / dt)

    for step in range(steps):
        signals = stream.get_signals(engine.t, dt)
        engine.step(signals)

        if step % 200 == 0:
            obs = engine.observe()
            print(
                f"t={engine.t:6.2f}s | N={obs['n_oscillators']:3d} | M={obs['n_carriers']:2d} | "
                f"nnz={obs['n_bonds']:4d} | R={obs['global_sync_R']:.3f} | "
                f"T_med={obs['medium_temperature']:.2f} | R_med={obs['medium_resource']:.2f}"
            )

    print()
    print(f"Final: {engine.oscillators.n} oscillators, {engine.carriers.m} carriers")
    print(f"Births: {len(engine.events.births)}, Deaths: {len(engine.events.deaths)}, Mitoses: {len(engine.events.mitoses)}")


if __name__ == "__main__":
    main()
