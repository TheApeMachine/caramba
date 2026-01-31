from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import torch
from tensordict import TensorDict

DTYPE_REAL = torch.float32
TAU = 2.0 * math.pi


# ============================================================
# Config
# ============================================================

@dataclass
class ManifoldConfig:
    # Integration step in seconds (used for phase advance + energy drains).
    dt: float = 0.005

    # Numerical stability
    eps: float = 1e-8

    # Energetics:
    # Bonds and oscillators spend energy each step; spent energy becomes heat.
    # This is the main global scale knob.
    hold_cost_scale: float = 5.0

    # Heat diffusion stability cap
    max_diffusion_alpha: float = 0.25

    # Gate width clamp (radians)
    gate_min: float = 0.05
    gate_max: float = 1.5

    # One mitosis per step (stability)
    one_split_per_step: bool = True

    # Optional: one merge per step (if merge enabled)
    one_merge_per_step: bool = True

    # Hard cap (safety) on emergent carriers
    max_carriers: int = 64

    # Top-down feedback smoothing (seconds). 0 => immediate.
    top_down_tau: float = 0.0

    # Internal feedback loop (off by default)
    internal_feedback_enabled: bool = False

    # Synthesis mode: if True, carriers drive oscillators (top-down generation)
    # If False, oscillators drive carriers (bottom-up analysis)
    synthesis_mode: bool = False

    # Restoring force strength (for synthesis mode)
    restoring_force_strength: float = 1.0

    # Mask sharpening. (Used by tf_mask softmax scaling)
    # Higher = more winner-take-most, less 50/50 splits when carriers are close.
    mask_sharpening: float = 5.0

    # Performance: wrap step/observe in torch.no_grad by default (recommended)
    no_grad: bool = True

    # Optional structural dynamics (off by default for backward compatibility)
    enable_template_relaxation: bool = False  # carriers drift toward bonded mass (analysis mode)
    enable_merge: bool = False                # merge adjacent carriers (analysis mode)

    # Optional attraction shaping (off by default)
    enable_harmonic_attraction: bool = False  # useful in tonal domains (audio)


# ============================================================
# Helpers
# ============================================================

def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """Map angle differences to [-pi, pi]."""
    x = x % TAU
    return torch.where(x > math.pi, x - TAU, x)


def gate_high(delta_phi: torch.Tensor, gate_width: torch.Tensor) -> torch.Tensor:
    """
    Hard pulse: 1 when |delta_phi| <= gate_width/2 else 0.
    Shapes must broadcast.
    """
    return (delta_phi.abs() <= (gate_width / 2.0)).to(DTYPE_REAL)


def safe_tanh(x: torch.Tensor) -> torch.Tensor:
    """Saturating clamp to [-1, 1] without choosing a max-abs constant."""
    return torch.tanh(x)


def _log_freq(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log(x.abs() + eps)


# ============================================================
# Observer / Feedback interfaces
# ============================================================

@runtime_checkable
class OutputObserver(Protocol):
    """
    Output observers augment observations. They should be side-effect free.

    observe(...)
      - manifold: current system
      - obs: base observation dict (already computed)
      - context: optional per-step context (e.g. current STFT frame info)
    """
    def observe(self, manifold: "Manifold", obs: dict, context: Optional[dict] = None) -> dict: ...


@runtime_checkable
class FeedbackController(Protocol):
    """
    Feedback controllers can create top-down feedback (external or internal).
    They may be used for "always-on" closed-loop behavior later.

    feedback(...) should return a dict compatible with Manifold.provide_feedback().
    Return None for no feedback.
    """
    def feedback(self, manifold: "Manifold", obs: dict, context: Optional[dict] = None) -> Optional[dict]: ...


# ============================================================
# Manifold
# ============================================================

class Manifold:
    """
    A dynamical system that can operate in two modes:

    **ANALYZER MODE (default):** Bottom-up signal separation / clustering
    - Oscillators = streaming input atoms (fixed data)
    - Carriers = templates that capture/bind atoms and can optionally adapt
    - Result: clustering, masking, analysis

    **SYNTHESIZER MODE:** Top-down signal generation
    - Carriers = fixed patterns (learned concepts/weights)
    - Oscillators = move to fit carriers (generated data)
    - Result: diffusion-style denoising / generation

    Notes:
      - This implementation is still fundamentally a 1D "spectral coordinate" system
        (the `frequency` field). You can reuse the same mechanics for other 1D
        latent coordinates by treating "frequency" as a generic coordinate.
      - For higher-dimensional latents, attach vectors to carriers (e.g. `state_vector`)
        and use `predict_next_token` / observers, or extend attraction accordingly.
    """

    def __init__(self, config: ManifoldConfig | None = None, device: torch.device | None = None):
        self.config = config or ManifoldConfig()
        self.device = device or torch.device("cpu")
        self.t = 0.0

        # External output observers (dashboard, logging, etc.)
        self.output_observers: list[OutputObserver] = []

        # Feedback controllers (external or internal policies)
        self.feedback_controllers: list[FeedbackController] = []

        # Top-down bias (raw, unbounded); applied as tanh(raw) in [-1,1]
        self._top_down_bias_raw = torch.empty(0, dtype=DTYPE_REAL, device=self.device)

        # Carrier identity
        self._next_carrier_id: int = 0
        self._carrier_names: list[str] = []

        # State
        self.state = TensorDict(
            {
                "oscillators": TensorDict(
                    {
                        "frequency": torch.empty(0, dtype=DTYPE_REAL, device=self.device),  # Hz / generic coord
                        "amplitude": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "phase": torch.empty(0, dtype=DTYPE_REAL, device=self.device),      # rad
                        "energy": torch.empty(0, dtype=DTYPE_REAL, device=self.device),     # >=0
                        "ttl": torch.empty(0, dtype=DTYPE_REAL, device=self.device),        # seconds remaining
                    },
                    batch_size=[0],
                ),
                "carriers": TensorDict(
                    {
                        "id": torch.empty(0, dtype=torch.int64, device=self.device),
                        "frequency": torch.empty(0, dtype=DTYPE_REAL, device=self.device),  # Hz / generic coord
                        "phase": torch.empty(0, dtype=DTYPE_REAL, device=self.device),      # rad
                        "base_width": torch.empty(0, dtype=DTYPE_REAL, device=self.device), # rad
                        "gate_width": torch.empty(0, dtype=DTYPE_REAL, device=self.device), # rad (emergent)
                        "heat": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "temperature": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "excitation": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "coherence": torch.empty(0, dtype=DTYPE_REAL, device=self.device),  # [0,1]
                        "energy": torch.empty(0, dtype=DTYPE_REAL, device=self.device),     # |phasor sum|
                    },
                    batch_size=[0],
                ),
                "bonds": TensorDict(
                    {
                        "presence": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),  # [Nosc,Ncar] in {0,1}
                        "energy": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),    # [Nosc,Ncar] >=0
                    },
                    batch_size=[],
                ),
            }
        )

    # ============================================================
    # Core step
    # ============================================================

    def step(self, signals: Sequence[Mapping[str, float]] | TensorDict, context: Optional[dict] = None) -> None:
        if self.config.no_grad:
            with torch.no_grad():
                self._step_impl(signals, context=context)
        else:
            self._step_impl(signals, context=context)

    def _step_impl(self, signals: Sequence[Mapping[str, float]] | TensorDict, context: Optional[dict] = None) -> None:
        dt = float(self.config.dt)

        # 0) Age out oscillators (expire at start so they don't participate forever)
        self._decay_oscillators(dt)

        # 1) Ingest
        new_inputs = self._batch_signals(signals)
        injected_energy = self._estimate_input_energy(new_inputs)

        # 2) Genesis
        if self.state.get("carriers").shape[0] == 0 and new_inputs.shape[0] > 0:
            self._genesis_from_signal(new_inputs, injected_energy)

        # 3) Append new oscillators
        n_new = int(new_inputs.shape[0])
        if n_new > 0:
            self._append_oscillators(new_inputs)

        # Nothing to do without carriers
        if self.state.get("carriers").shape[0] == 0:
            self.t += dt
            return

        # 4) Bond new oscillators
        if n_new > 0:
            self._bond_new_oscillators(new_osc_count=n_new)

        # 5) Update carrier coherence/energy (for scoring & dashboards)
        self._update_carrier_coherence_energy()

        # 6) Optional: relax carriers toward bonded mass (analysis mode)
        if (not self.config.synthesis_mode) and self.config.enable_template_relaxation:
            self._relax_carrier_templates()

        # 7) Temperature/excitation/gate
        self._update_temperature_excitation_gate()

        # 8) Costs -> heat
        self._drain_bonds_to_heat()
        self._drain_oscillators_to_heat()

        # 9) Diffuse heat
        self._diffuse_heat()

        # 10) Recompute temperature/excitation/gate after heat changes
        self._update_temperature_excitation_gate()

        # 11) Mitosis (analysis mode)
        if not self.config.synthesis_mode:
            self._spectral_mitosis_option_a()
            if self.config.enable_merge:
                self._merge_adjacent_carriers()

        # 12) Advance phases
        self._advance_phases(dt)

        # 12.5) Synthesis mode: apply restoring force (top-down generation)
        if self.config.synthesis_mode:
            self.diffuse_step(dt, strength=self.config.restoring_force_strength, noise_scale=10.0)

        # 13) Internal feedback loop (optional)
        if self.config.internal_feedback_enabled:
            self._run_feedback_controllers(context=context)

        self.t += dt

    # ============================================================
    # Observation Interface (canonical + observers)
    # ============================================================

    def observe(
        self,
        feedback: Optional[dict] = None,
        *,
        full_matrices: bool = True,
        context: Optional[dict] = None,
        run_observers: bool = True,
    ) -> dict:
        """
        Canonical observation interface.

        - Applies top-down feedback (if provided)
        - Returns a rich dict containing:
            global metrics, emergent "medium" scalars,
            carrier-level tensors, oscillator-level tensors,
            and optionally full matrices (tuning, attraction, soft assignment, presence, bond_energy)
        - Augments with any registered output_observers.
        """
        if self.config.no_grad:
            with torch.no_grad():
                return self._observe_impl(feedback, full_matrices=full_matrices, context=context, run_observers=run_observers)
        return self._observe_impl(feedback, full_matrices=full_matrices, context=context, run_observers=run_observers)

    def _observe_impl(
        self,
        feedback: Optional[dict],
        *,
        full_matrices: bool,
        context: Optional[dict],
        run_observers: bool,
    ) -> dict:
        self.provide_feedback(feedback)

        obs = self._observe_base(full_matrices=full_matrices)

        if run_observers:
            for ob in self.output_observers:
                try:
                    extra = ob.observe(self, obs, context=context)
                    if isinstance(extra, dict) and extra:
                        obs.update(extra)
                except Exception as e:
                    obs.setdefault("observer_errors", []).append((type(ob).__name__, str(e)))

        return obs

    def provide_feedback(self, feedback: Optional[dict]) -> None:
        """
        Store bounded top-down feedback for later dynamics updates.

        Supported:
          feedback = {
            "utility": float (global multiplier, optional),
            "carrier_utility": dict[id_or_index -> float] | Tensor[m] | Sequence[float]
          }

        This updates self._top_down_bias_raw (one value per carrier).
        Applied during bonding as an additive bias in log-attraction space.
        """
        if feedback is None:
            return

        carriers = self.state.get("carriers")
        m = int(carriers.shape[0])
        if m == 0:
            return

        cu = feedback.get("carrier_utility", None)
        if cu is None:
            return

        # Ensure bias vector exists
        if self._top_down_bias_raw.numel() != m:
            self._top_down_bias_raw = torch.zeros(m, dtype=DTYPE_REAL, device=self.device)

        # Parse carrier utility into vector u[m]
        u = torch.zeros(m, dtype=DTYPE_REAL, device=self.device)

        ids = carriers.get("id")
        id_to_idx = {int(ids[i].item()): i for i in range(m)}

        if isinstance(cu, dict):
            for k, v in cu.items():
                key = k
                if isinstance(key, str):
                    try:
                        key = int(key)
                    except Exception:
                        pass
                if isinstance(key, int) and key in id_to_idx:
                    u[id_to_idx[key]] = float(v)
                elif isinstance(key, int) and 0 <= key < m:
                    # allow index addressing too
                    u[key] = float(v)
        elif isinstance(cu, torch.Tensor):
            cu_t = cu.to(device=self.device, dtype=DTYPE_REAL).flatten()
            if cu_t.numel() != m:
                return
            u = cu_t
        else:
            try:
                seq = list(cu)  # type: ignore[arg-type]
            except Exception:
                return
            if len(seq) != m:
                return
            u = torch.tensor(seq, dtype=DTYPE_REAL, device=self.device)

        g = feedback.get("utility", 1.0)
        try:
            g = float(g)
        except Exception:
            g = 1.0
        u = u * g

        # Smooth update
        tau = float(self.config.top_down_tau)
        if tau > 0.0:
            alpha = float(self.config.dt) / tau
            alpha = max(0.0, min(1.0, alpha))
        else:
            alpha = 1.0

        self._top_down_bias_raw = (1.0 - alpha) * self._top_down_bias_raw + alpha * u

    def observe_carriers_only(self) -> torch.Tensor:
        """
        Compact carrier features tensor [M, 9].
        """
        obs = self.observe(full_matrices=False, run_observers=False)
        m = int(obs["n_carriers"])
        if m == 0:
            return torch.empty(0, 9, dtype=DTYPE_REAL, device=self.device)

        energy = obs["carrier_energy"]
        phase = obs["carrier_phase"]
        freq = obs["carrier_frequency"]  # Hz / coord
        temp = obs["carrier_temperature"]
        coh = obs["carrier_coherence"]
        gate = obs["carrier_gate_width"] / math.pi
        heat = obs["carrier_heat"]
        bias = obs["top_down_carrier_bias"]

        features = torch.stack(
            [
                energy,
                torch.sin(phase),
                torch.cos(phase),
                freq,
                temp,
                coh,
                gate,
                heat,
                bias,
            ],
            dim=1,
        ).to(dtype=DTYPE_REAL, device=self.device)
        return features

    def observe_global(self) -> torch.Tensor:
        """
        Compact global features tensor [8].
        """
        obs = self.observe(full_matrices=False, run_observers=False)

        n_car = float(obs["n_carriers"])
        n_osc = float(obs["n_oscillators_live"])
        R = float(obs["global_sync_R"])
        L = float(obs["L_comp"])

        if obs["n_carriers"] > 0:
            energies = obs["carrier_energy"]
            mean_energy = float(energies.mean().item())
            mean_coh = float(obs["carrier_coherence"].mean().item())

            probs = energies / (energies.sum() + self.config.eps)
            energy_entropy = float((-(probs * torch.log(probs + self.config.eps))).sum().item())

            centers = obs["carrier_spectral_center_hz"]
            spectral_coverage = float(((centers.max() - centers.min()) / (centers.mean() + self.config.eps)).item())
        else:
            mean_energy = 0.0
            mean_coh = 0.0
            energy_entropy = 0.0
            spectral_coverage = 0.0

        return torch.tensor(
            [n_car, n_osc, R, L, mean_energy, mean_coh, energy_entropy, spectral_coverage],
            dtype=DTYPE_REAL,
            device=self.device,
        )

    # ============================================================
    # Internal / base observation computation
    # ============================================================

    def _observe_base(self, *, full_matrices: bool) -> dict:
        osc = self.state.get("oscillators")
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")

        n_car = int(carriers.shape[0])
        n_osc_total = int(osc.shape[0])

        # Define "live" oscillators as ttl>0 and amplitude>0 (or bonded)
        if n_osc_total > 0:
            ttl = osc.get("ttl")
            amp = osc.get("amplitude")
            P = bonds.get("presence")
            bonded = (P.sum(dim=1) > 0.0) if P.numel() > 0 else torch.zeros(n_osc_total, dtype=torch.bool, device=self.device)
            live_mask = (ttl > 0.0) & ((amp > self.config.eps) | bonded)
        else:
            live_mask = torch.zeros(0, dtype=torch.bool, device=self.device)

        n_osc_live = int(live_mask.sum().item()) if live_mask.numel() > 0 else 0

        n_bonds = self.nnz_P()
        R = self.global_sync_R()
        L = self.L_comp()

        # "Medium" scalars
        if n_car > 0:
            mean_temp = float(carriers.get("temperature").mean().item())
            total_heat = float(carriers.get("heat").sum().item())
        else:
            mean_temp = 0.0
            total_heat = 0.0

        # Resource: total stored energy in oscillators + bonds + heat
        total_osc_energy = float(osc.get("energy").sum().item()) if n_osc_total > 0 else 0.0
        total_bond_energy = float(bonds.get("energy").sum().item()) if bonds.get("energy").numel() > 0 else 0.0
        total_resource = total_osc_energy + total_bond_energy + total_heat

        # Carrier identity
        carrier_ids = carriers.get("id").tolist() if n_car > 0 else []
        carrier_names = list(self._carrier_names)

        # Ensure top-down bias vector matches carriers
        if self._top_down_bias_raw.numel() != n_car:
            self._top_down_bias_raw = torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device)
        bias = safe_tanh(self._top_down_bias_raw)

        # Carrier-level tensors
        if n_car > 0:
            carrier_freq = carriers.get("frequency").clone()
            carrier_phase = carriers.get("phase").clone()
            carrier_gate = carriers.get("gate_width").clone()
            carrier_temp = carriers.get("temperature").clone()
            carrier_heat = carriers.get("heat").clone()
            carrier_exc = carriers.get("excitation").clone()
            carrier_coh = carriers.get("coherence").clone()
            carrier_eng = carriers.get("energy").clone()
        else:
            carrier_freq = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_phase = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_gate = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_temp = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_heat = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_exc = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_coh = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            carrier_eng = torch.empty(0, dtype=DTYPE_REAL, device=self.device)

        # Oscillator-level tensors (live-only view for dashboards)
        if n_osc_total > 0 and n_osc_live > 0:
            osc_freq = osc.get("frequency")[live_mask].clone()
            osc_phase = osc.get("phase")[live_mask].clone()
            osc_amp = osc.get("amplitude")[live_mask].clone()
            osc_energy = osc.get("energy")[live_mask].clone()
            osc_ttl = osc.get("ttl")[live_mask].clone()
        else:
            osc_freq = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            osc_phase = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            osc_amp = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            osc_energy = torch.empty(0, dtype=DTYPE_REAL, device=self.device)
            osc_ttl = torch.empty(0, dtype=DTYPE_REAL, device=self.device)

        obs: dict[str, Any] = {
            "t": float(self.t),
            "n_oscillators_total": n_osc_total,
            "n_oscillators_live": n_osc_live,
            "n_carriers": n_car,
            "n_bonds": n_bonds,
            "global_sync_R": float(R),
            "L_comp": int(L),

            # Emergent medium snapshot
            "medium_temperature": float(mean_temp),
            "medium_heat": float(total_heat),
            "medium_resource": float(total_resource),

            # Physics/config snapshot (small)
            "physics": {
                "dt": float(self.config.dt),
                "hold_cost_scale": float(self.config.hold_cost_scale),
                "gate_min": float(self.config.gate_min),
                "gate_max": float(self.config.gate_max),
                "top_down_tau": float(self.config.top_down_tau),
                "internal_feedback_enabled": bool(self.config.internal_feedback_enabled),
                "synthesis_mode": bool(self.config.synthesis_mode),
            },

            # Carrier-level
            "carrier_ids": [int(x) for x in carrier_ids],
            "carrier_names": carrier_names,
            "carrier_frequency": carrier_freq,
            "carrier_phase": carrier_phase,
            "carrier_gate_width": carrier_gate,
            "carrier_temperature": carrier_temp,
            "carrier_heat": carrier_heat,
            "carrier_excitation": carrier_exc,
            "carrier_coherence": carrier_coh,
            "carrier_energy": carrier_eng,
            "top_down_carrier_bias": bias.clone(),

            # Oscillator-level (live only)
            "osc_frequency": osc_freq,
            "osc_phase": osc_phase,
            "osc_amplitude": osc_amp,
            "osc_energy": osc_energy,
            "osc_ttl": osc_ttl,
        }

        # Spectral profiles (carrier-level) from current live oscillators + current bonds (live rows)
        center_hz, var_hz2, multimodal = self._compute_carrier_spectral_profiles(live_mask=live_mask)
        obs["carrier_spectral_center_hz"] = center_hz
        obs["carrier_spectral_variance_hz2"] = var_hz2
        obs["carrier_is_multimodal"] = multimodal

        if not full_matrices:
            return obs

        # Full matrices (live oscillators only for sanity)
        if n_car == 0 or n_osc_live == 0:
            obs.update(
                {
                    "presence": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),
                    "bond_energy": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),
                    "tuning": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),
                    "attraction": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),
                    "soft_assignment": torch.empty(0, 0, dtype=DTYPE_REAL, device=self.device),
                }
            )
            return obs

        P_full = bonds.get("presence")
        B_full = bonds.get("energy")
        P_live = P_full[live_mask] if P_full.numel() > 0 else torch.empty(0, n_car, dtype=DTYPE_REAL, device=self.device)
        B_live = B_full[live_mask] if B_full.numel() > 0 else torch.empty(0, n_car, dtype=DTYPE_REAL, device=self.device)

        # Tuning (pulse-open indicator for each osc-carrier pair)
        dphi = wrap_pi(osc_phase.unsqueeze(1) - carrier_phase.unsqueeze(0))
        tuning = gate_high(dphi, carrier_gate.unsqueeze(0))

        # Attraction [Nlive, M]
        attraction = self._attraction_matrix(osc_freq, osc_phase)

        # Soft assignment from attraction + top-down bias (log space)
        bias_row = safe_tanh(self._top_down_bias_raw).unsqueeze(0)  # [1,M]
        scores = torch.log(attraction + self.config.eps) + bias_row
        soft = torch.softmax(scores, dim=1)

        obs.update(
            {
                "presence": P_live,
                "bond_energy": B_live,
                "tuning": tuning,
                "attraction": attraction,
                "soft_assignment": soft,
            }
        )
        return obs

    # ============================================================
    # Signal ingestion
    # ============================================================

    def _batch_signals(self, signals: Sequence[Mapping[str, float]] | TensorDict) -> TensorDict:
        if isinstance(signals, TensorDict):
            # Expect keys: frequency, amplitude, phase, duration
            if signals.shape[0] == 0:
                return TensorDict(
                    {
                        "frequency": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "amplitude": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "phase": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "duration": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                    },
                    batch_size=[0],
                )
            return signals.to(device=self.device)

        n = len(signals)
        if n == 0:
            return TensorDict(
                {
                    "frequency": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                    "amplitude": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                    "phase": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                    "duration": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                },
                batch_size=[0],
            )

        freqs = torch.empty(n, dtype=DTYPE_REAL, device=self.device)
        amps = torch.empty(n, dtype=DTYPE_REAL, device=self.device)
        phases = torch.empty(n, dtype=DTYPE_REAL, device=self.device)
        durs = torch.empty(n, dtype=DTYPE_REAL, device=self.device)

        for i, s in enumerate(signals):
            freqs[i] = float(s["frequency"])
            amps[i] = float(s["amplitude"])
            phases[i] = float(s["phase"])
            durs[i] = float(s.get("duration", self.config.dt))

        return TensorDict({"frequency": freqs, "amplitude": amps, "phase": phases, "duration": durs}, batch_size=[n])

    def _estimate_input_energy(self, inputs: TensorDict) -> torch.Tensor:
        if inputs.shape[0] == 0:
            return torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
        # Unitless injected energy
        return (inputs.get("amplitude") * inputs.get("frequency").abs()).sum()

    # ============================================================
    # Genesis / append
    # ============================================================

    def _genesis_from_signal(self, inputs: TensorDict, injected_energy: torch.Tensor) -> None:
        # Get all inputs
        freqs = inputs.get("frequency")
        amps = inputs.get("amplitude")
        phases = inputs.get("phase")

        # Sort by amplitude to find the loudest peaks - spawn multiple carriers at once
        n_spawns = min(3, int(freqs.shape[0]))  # Spawn up to 3 carriers at once
        if n_spawns == 0:
            return

        # Get indices of top amplitudes
        _, top_indices = torch.topk(amps, n_spawns)

        # Start narrow (0.1 rad) so they don't overlap
        base_w_val = 0.1
        base_w = torch.full((n_spawns,), base_w_val, dtype=DTYPE_REAL, device=self.device)

        # Collect IDs and build tensors
        carrier_ids = []
        for _ in range(n_spawns):
            cid = self._next_carrier_id
            self._next_carrier_id += 1
            self._carrier_names.append(f"C{cid}")
            carrier_ids.append(cid)

        selected_freqs = freqs[top_indices]
        selected_phases = phases[top_indices]
        split_heat = (injected_energy / n_spawns).expand(n_spawns).to(DTYPE_REAL)

        carriers = TensorDict(
            {
                "id": torch.tensor(carrier_ids, dtype=torch.int64, device=self.device),
                "frequency": selected_freqs.to(DTYPE_REAL),
                "phase": selected_phases.to(DTYPE_REAL),
                "base_width": base_w,
                "gate_width": base_w.clone(),
                "heat": split_heat,
                "temperature": torch.zeros(n_spawns, dtype=DTYPE_REAL, device=self.device),
                "excitation": torch.zeros(n_spawns, dtype=DTYPE_REAL, device=self.device),
                "coherence": torch.zeros(n_spawns, dtype=DTYPE_REAL, device=self.device),
                "energy": torch.zeros(n_spawns, dtype=DTYPE_REAL, device=self.device),
            },
            batch_size=[n_spawns],
        )
        self.state.set("carriers", carriers)

        # Initialize bonds for N carriers
        self.state.set(
            "bonds",
            TensorDict(
                {
                    "presence": torch.empty(0, n_spawns, dtype=DTYPE_REAL, device=self.device),
                    "energy": torch.empty(0, n_spawns, dtype=DTYPE_REAL, device=self.device),
                },
                batch_size=[],
            ),
        )

        # Initialize bias vector for N carriers
        self._top_down_bias_raw = torch.zeros(n_spawns, dtype=DTYPE_REAL, device=self.device)

    def _append_oscillators(self, new_osc: TensorDict) -> None:
        osc = self.state.get("oscillators")
        n_old = int(osc.shape[0])
        n_new = int(new_osc.shape[0])

        # Initial oscillator energy (unitless)
        osc_energy = (new_osc.get("amplitude") * new_osc.get("frequency").abs()).to(DTYPE_REAL)
        ttl = new_osc.get("duration").to(DTYPE_REAL).clamp(min=float(self.config.dt))  # ensure >= dt

        self.state.set(
            "oscillators",
            TensorDict(
                {
                    "frequency": torch.cat([osc.get("frequency"), new_osc.get("frequency")], dim=0),
                    "amplitude": torch.cat([osc.get("amplitude"), new_osc.get("amplitude")], dim=0),
                    "phase": torch.cat([osc.get("phase"), new_osc.get("phase")], dim=0),
                    "energy": torch.cat([osc.get("energy"), osc_energy], dim=0),
                    "ttl": torch.cat([osc.get("ttl"), ttl], dim=0),
                },
                batch_size=[n_old + n_new],
            ),
        )

        # Resize bonds: add rows
        bonds = self.state.get("bonds")
        P = bonds.get("presence")
        B = bonds.get("energy")
        n_car = int(self.state.get("carriers").shape[0])

        if P.numel() == 0:
            P = torch.zeros(n_old + n_new, n_car, dtype=DTYPE_REAL, device=self.device)
            B = torch.zeros(n_old + n_new, n_car, dtype=DTYPE_REAL, device=self.device)
        else:
            padP = torch.zeros(n_new, n_car, dtype=P.dtype, device=P.device)
            padB = torch.zeros(n_new, n_car, dtype=B.dtype, device=B.device)
            P = torch.cat([P, padP], dim=0)
            B = torch.cat([B, padB], dim=0)

        self.state.set("bonds", TensorDict({"presence": P, "energy": B}, batch_size=[]))

    # ============================================================
    # Oscillator lifecycle
    # ============================================================

    def _decay_oscillators(self, dt: float) -> None:
        """
        Decrease TTL; when expired, zero its amplitude/energy and remove bonds (presence/energy rows -> 0).
        This keeps the system from accumulating unbounded oscillators in streaming settings.
        """
        osc = self.state.get("oscillators")
        n = int(osc.shape[0])
        if n == 0:
            return

        ttl = osc.get("ttl") - dt
        alive = ttl > 0.0

        # If nothing expired, just update ttl and return
        if alive.all():
            osc.set("ttl", ttl)
            self.state.set("oscillators", osc)
            return

        # Expire
        amp = osc.get("amplitude")
        eng = osc.get("energy")

        amp = torch.where(alive, amp, torch.zeros_like(amp))
        eng = torch.where(alive, eng, torch.zeros_like(eng))
        ttl = torch.where(alive, ttl, torch.zeros_like(ttl))

        osc.set("amplitude", amp)
        osc.set("energy", eng)
        osc.set("ttl", ttl)
        self.state.set("oscillators", osc)

        # Remove bonds for expired oscillators
        bonds = self.state.get("bonds")
        P = bonds.get("presence")
        B = bonds.get("energy")
        if P.numel() > 0:
            P = P.clone()
            B = B.clone()
            dead_idx = torch.nonzero(~alive, as_tuple=False).squeeze(1)
            if dead_idx.numel() > 0:
                P[dead_idx, :] = 0.0
                B[dead_idx, :] = 0.0
            bonds.set("presence", P)
            bonds.set("energy", B)
            self.state.set("bonds", bonds)

    def compact_dead_oscillators(self) -> None:
        """
        Physically remove dead oscillators (ttl<=0 and no bonds). This is optional and expensive.
        Useful for long offline runs to keep tensors small.
        """
        osc = self.state.get("oscillators")
        bonds = self.state.get("bonds")
        n = int(osc.shape[0])
        if n == 0:
            return

        ttl = osc.get("ttl")
        P = bonds.get("presence")
        has_bond = (P.sum(dim=1) > 0.0) if P.numel() > 0 else torch.zeros(n, dtype=torch.bool, device=self.device)
        keep = (ttl > 0.0) | has_bond

        if keep.all():
            return

        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            # Reset to empty
            self.state.set(
                "oscillators",
                TensorDict(
                    {
                        "frequency": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "amplitude": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "phase": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "energy": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                        "ttl": torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                    },
                    batch_size=[0],
                ),
            )
            self.state.set(
                "bonds",
                TensorDict(
                    {
                        "presence": torch.empty(0, int(self.state.get("carriers").shape[0]), dtype=DTYPE_REAL, device=self.device),
                        "energy": torch.empty(0, int(self.state.get("carriers").shape[0]), dtype=DTYPE_REAL, device=self.device),
                    },
                    batch_size=[],
                ),
            )
            return

        # Reindex oscillators
        new_osc = TensorDict(
            {
                "frequency": osc.get("frequency")[idx],
                "amplitude": osc.get("amplitude")[idx],
                "phase": osc.get("phase")[idx],
                "energy": osc.get("energy")[idx],
                "ttl": osc.get("ttl")[idx],
            },
            batch_size=[int(idx.numel())],
        )
        self.state.set("oscillators", new_osc)

        # Reindex bond rows
        if P.numel() > 0:
            new_P = P[idx]
            new_B = bonds.get("energy")[idx]
        else:
            m = int(self.state.get("carriers").shape[0])
            new_P = torch.empty(int(idx.numel()), m, dtype=DTYPE_REAL, device=self.device)
            new_B = torch.empty(int(idx.numel()), m, dtype=DTYPE_REAL, device=self.device)

        self.state.set("bonds", TensorDict({"presence": new_P, "energy": new_B}, batch_size=[]))

    # ============================================================
    # Attraction / tuning / masks
    # ============================================================

    def _attraction_matrix(self, osc_freq_hz: torch.Tensor, osc_phase_rad: torch.Tensor) -> torch.Tensor:
        """
        A_{ik} = duty_k * exp(-(rel^2)/(duty^2))

        rel = (f_i - f_k) / (|f_k| + eps)
        duty = gate_width / 2π in (0,1]

        Phase alignment remains disabled by default (see config.enable_harmonic_attraction for optional shaping).
        """
        carriers = self.state.get("carriers")
        f_k = carriers.get("frequency")  # [M]
        w_k = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)  # [M]

        duty = (w_k / TAU).clamp(min=self.config.eps, max=1.0)  # [M]
        scale = duty.unsqueeze(0) ** 2 + self.config.eps        # [1,M]

        # Relative frequency alignment
        rel = (osc_freq_hz.unsqueeze(1) - f_k.unsqueeze(0)) / (f_k.abs().unsqueeze(0) + self.config.eps)  # [N,M]
        base = torch.exp(-(rel * rel) / scale)

        if self.config.enable_harmonic_attraction:
            # Optional: harmonic attraction around integer ratios
            ratio = osc_freq_hz.unsqueeze(1) / (f_k.abs().unsqueeze(0) + self.config.eps)
            h = torch.round(ratio).clamp(min=1.0)
            herr = ratio - h
            harm = torch.exp(-(herr * herr) / scale)
            # Blend: when duty is small (narrow), allow more harmonic influence
            w_h = (1.0 - duty).unsqueeze(0).clamp(0.0, 1.0)
            base = (1.0 - w_h) * base + w_h * harm

        A = duty.unsqueeze(0) * base
        return A.to(DTYPE_REAL)

    def tuning_matrix(self, osc_phase_rad: torch.Tensor) -> torch.Tensor:
        """
        T_{ik} in {0,1}: carrier k is listening to oscillator i (phase window open).
        """
        carriers = self.state.get("carriers")
        if carriers.shape[0] == 0 or osc_phase_rad.numel() == 0:
            return torch.empty(int(osc_phase_rad.shape[0]), int(carriers.shape[0]), dtype=DTYPE_REAL, device=self.device)

        phi_k = carriers.get("phase")
        w_k = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)

        dphi = wrap_pi(osc_phase_rad.unsqueeze(1) - phi_k.unsqueeze(0))
        return gate_high(dphi, w_k.unsqueeze(0))

    def tf_mask(self, bin_freq_hz: torch.Tensor, bin_phase_rad: torch.Tensor) -> torch.Tensor:
        """
        Soft time-frequency assignment mask for an arbitrary set of bins (e.g., one STFT frame).

        Returns:
          mask: [F, M] where sum_k mask[f,k] = 1

        This is the core ingredient for audio separation (masking STFT bins per carrier).
        """
        carriers = self.state.get("carriers")
        m = int(carriers.shape[0])
        f = int(bin_freq_hz.shape[0])
        if m == 0 or f == 0:
            return torch.empty(f, m, dtype=DTYPE_REAL, device=self.device)

        A = self._attraction_matrix(bin_freq_hz, bin_phase_rad)  # [F,M]

        # Apply top-down bias (log-attraction space)
        if self._top_down_bias_raw.numel() != m:
            self._top_down_bias_raw = torch.zeros(m, dtype=DTYPE_REAL, device=self.device)
        bias = safe_tanh(self._top_down_bias_raw).unsqueeze(0)  # [1,M]

        scores = torch.log(A + self.config.eps) + bias

        sharpening = float(self.config.mask_sharpening)
        if sharpening <= 0.0:
            sharpening = 1.0

        return torch.softmax(scores * sharpening, dim=1).to(DTYPE_REAL)

    # ============================================================
    # Bonding
    # ============================================================

    def _bond_new_oscillators(self, new_osc_count: int) -> None:
        osc = self.state.get("oscillators")
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")

        P = bonds.get("presence")
        B = bonds.get("energy")

        n_osc = int(osc.shape[0])
        n_car = int(carriers.shape[0])
        n_old = n_osc - new_osc_count
        if n_car == 0 or new_osc_count <= 0:
            return

        # Ensure bias vector matches carriers
        if self._top_down_bias_raw.numel() != n_car:
            self._top_down_bias_raw = torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device)

        # New oscillator views
        new_freq = osc.get("frequency")[n_old:]
        new_phase = osc.get("phase")[n_old:]
        new_amp = osc.get("amplitude")[n_old:]
        new_energy = osc.get("energy")[n_old:]

        # Attraction to carriers [new, M]
        A_new = self._attraction_matrix(new_freq, new_phase)

        # Top-down bias in log space for hard choice
        bias = safe_tanh(self._top_down_bias_raw).unsqueeze(0)  # [1,M]
        scores = torch.log(A_new + self.config.eps) + bias
        k_star = torch.argmax(scores, dim=1)  # [new]
        a_star = A_new.gather(1, k_star.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)  # [new]

        # Presence one-hot
        P_new = torch.zeros(new_osc_count, n_car, dtype=DTYPE_REAL, device=self.device)
        P_new.scatter_(1, k_star.unsqueeze(1), 1.0)
        P[n_old:] = P_new

        # (1) Capture: oscillator energy -> bond energy (conservative)
        captured = (new_energy * a_star).clamp(min=0.0)
        new_energy = (new_energy - captured).clamp(min=0.0)
        bond_budget = captured.clone()

        # (2) Sympathetic transfer: old oscillators spend energy -> new bonds
        if n_old > 0:
            old_freq = osc.get("frequency")[:n_old]
            old_phase = osc.get("phase")[:n_old]
            old_amp = osc.get("amplitude")[:n_old]
            old_energy = osc.get("energy")[:n_old]

            car_phase = carriers.get("phase")
            car_freq = carriers.get("frequency")
            car_gate = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)
            duty = (car_gate / TAU).clamp(min=self.config.eps, max=1.0)  # [M]

            for k in range(n_car):
                idx_new_k = torch.nonzero(k_star == k, as_tuple=False).squeeze(1)
                if idx_new_k.numel() == 0:
                    continue

                # Old oscillators bonded to k
                bonded_old = P[:n_old, k] > 0.0
                if not bonded_old.any():
                    continue

                # Carrier must be listening to old oscillators (pulse-high)
                dphi_old = wrap_pi(old_phase - car_phase[k])
                g_old = gate_high(dphi_old, car_gate[k]) > 0.0
                idx_old_active = torch.nonzero(bonded_old & g_old, as_tuple=False).squeeze(1)
                if idx_old_active.numel() == 0:
                    continue

                # Also require the carrier to be listening when the new oscillators arrive
                dphi_new = wrap_pi(new_phase[idx_new_k] - car_phase[k])
                g_new = gate_high(dphi_new, car_gate[k])  # [m]
                if not (g_new > 0.0).any():
                    continue

                # Effective new amplitudes during listening
                ai = (new_amp[idx_new_k] * g_new).to(DTYPE_REAL)  # [m]
                if torch.count_nonzero(ai).item() == 0:
                    continue

                # Active old subset
                fj = old_freq[idx_old_active]
                phij = old_phase[idx_old_active]
                aj = old_amp[idx_old_active]
                Ej = old_energy[idx_old_active]

                # Phase kernel (dimensionless)
                K_phase = torch.cos(wrap_pi(new_phase[idx_new_k].unsqueeze(1) - phij.unsqueeze(0))).abs()  # [m,n]

                # Frequency kernel (dimensionless, duty-scaled)
                rel_ij = (new_freq[idx_new_k].unsqueeze(1) - fj.unsqueeze(0)) / (car_freq[k].abs() + self.config.eps)
                K_freq = torch.exp(-(rel_ij * rel_ij) / (duty[k] ** 2 + self.config.eps))  # [m,n]

                demand = (ai.unsqueeze(1) * aj.unsqueeze(0)) * K_phase * K_freq  # [m,n]
                demand_old = demand.sum(dim=0)  # [n]
                if torch.count_nonzero(demand_old).item() == 0:
                    continue

                spent_old = torch.minimum(demand_old, Ej)
                weights = demand / (demand_old.unsqueeze(0) + self.config.eps)  # [m,n]
                received = weights @ spent_old  # [m]

                bond_budget[idx_new_k] += received
                Ej_new = (Ej - spent_old).clamp(min=0.0)
                old_energy[idx_old_active] = Ej_new

            # Write back old energy
            osc_energy_all = osc.get("energy")
            osc_energy_all[:n_old] = old_energy
            osc.set("energy", osc_energy_all)

        # Write back new energy after capture
        osc_energy_all = osc.get("energy")
        osc_energy_all[n_old:] = new_energy
        osc.set("energy", osc_energy_all)
        self.state.set("oscillators", osc)

        # Write new bond energy
        B_new = torch.zeros(new_osc_count, n_car, dtype=DTYPE_REAL, device=self.device)
        B_new.scatter_(1, k_star.unsqueeze(1), bond_budget.unsqueeze(1))
        B[n_old:] = B_new

        self.state.set("bonds", TensorDict({"presence": P, "energy": B}, batch_size=[]))

    def rebond_all_oscillators(self, *, conservative: bool = False) -> None:
        """
        Re-bond all *live* oscillators (hard assignment), updating the full bond matrices.

        This is primarily used by synthesis/diffusion routines where oscillators move
        without being "new", but you still want bond metrics + carrier energy/coherence
        to reflect the current state.

        - If conservative=True, the captured bond energy is subtracted from oscillator energy.
        - If conservative=False (default), bond energy is written for metrics without changing oscillator energy.
        """
        osc = self.state.get("oscillators")
        carriers = self.state.get("carriers")
        n_osc = int(osc.shape[0])
        n_car = int(carriers.shape[0])
        if n_osc == 0 or n_car == 0:
            return

        ttl = osc.get("ttl")
        alive = ttl > 0.0
        if not alive.any():
            self.state.set(
                "bonds",
                TensorDict(
                    {
                        "presence": torch.zeros(n_osc, n_car, dtype=DTYPE_REAL, device=self.device),
                        "energy": torch.zeros(n_osc, n_car, dtype=DTYPE_REAL, device=self.device),
                    },
                    batch_size=[],
                ),
            )
            return

        f = osc.get("frequency")
        ph = osc.get("phase")
        E = osc.get("energy")

        A = self._attraction_matrix(f, ph)  # [N,M]

        if self._top_down_bias_raw.numel() != n_car:
            self._top_down_bias_raw = torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device)
        bias = safe_tanh(self._top_down_bias_raw).unsqueeze(0)

        scores = torch.log(A + self.config.eps) + bias
        k_star = torch.argmax(scores, dim=1)  # [N]
        a_star = A.gather(1, k_star.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)

        P = torch.zeros(n_osc, n_car, dtype=DTYPE_REAL, device=self.device)
        P.scatter_(1, k_star.unsqueeze(1), 1.0)
        P = P * alive.unsqueeze(1).to(DTYPE_REAL)

        bond_budget = (E * a_star).clamp(min=0.0)
        if conservative:
            E = (E - bond_budget).clamp(min=0.0)
            osc.set("energy", E)
            self.state.set("oscillators", osc)

        B = torch.zeros(n_osc, n_car, dtype=DTYPE_REAL, device=self.device)
        B.scatter_(1, k_star.unsqueeze(1), bond_budget.unsqueeze(1))
        B = B * alive.unsqueeze(1).to(DTYPE_REAL)

        self.state.set("bonds", TensorDict({"presence": P, "energy": B}, batch_size=[]))

    # ============================================================
    # Carrier coherence / energy (for scoring and viz)
    # ============================================================

    def _update_carrier_coherence_energy(self) -> None:
        """
        Vectorized coherence/energy update.

        For each carrier k:
          - compute phasor sum S_k = Σ_i w_i * exp(i*phi_i)
            where w_i = amplitude_i * presence_{ik} * gate_high(Δphi_{ik})
          - coherence_k = |S_k| / (Σ_i w_i)
          - energy_k = |S_k|
        """
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")
        osc = self.state.get("oscillators")

        n_car = int(carriers.shape[0])
        n_osc = int(osc.shape[0])
        if n_car == 0:
            return

        P = bonds.get("presence")
        if P.numel() == 0 or n_osc == 0:
            carriers.set("coherence", torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device))
            carriers.set("energy", torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device))
            self.state.set("carriers", carriers)
            return

        ttl = osc.get("ttl")
        alive = ttl > 0.0
        if not alive.any():
            carriers.set("coherence", torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device))
            carriers.set("energy", torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device))
            self.state.set("carriers", carriers)
            return

        osc_phase = osc.get("phase")
        osc_amp = osc.get("amplitude")

        car_phase = carriers.get("phase")
        car_gate = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)

        dphi = wrap_pi(osc_phase.unsqueeze(1) - car_phase.unsqueeze(0))  # [N,M]
        g = gate_high(dphi, car_gate.unsqueeze(0))  # [N,M]

        W = P * alive.unsqueeze(1).to(DTYPE_REAL) * osc_amp.unsqueeze(1) * g  # [N,M]
        wsum = W.sum(dim=0)  # [M]

        z = torch.exp(1j * osc_phase.to(torch.complex64))  # [N]
        S = (W.to(torch.complex64) * z.unsqueeze(1)).sum(dim=0)  # [M]
        mag = torch.abs(S).to(DTYPE_REAL)

        coh = torch.where(wsum > self.config.eps, (mag / (wsum + self.config.eps)).clamp(0.0, 1.0), torch.zeros_like(mag))

        carriers.set("coherence", coh)
        carriers.set("energy", mag)
        self.state.set("carriers", carriers)

    # ============================================================
    # Optional: relax templates (analysis)
    # ============================================================

    def _relax_carrier_templates(self) -> None:
        """
        Optional analyzer-mode template relaxation:
          carriers.frequency <- weighted mean of bonded oscillator frequencies (log domain, scale-free)

        This is off by default for backward compatibility.
        """
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")
        osc = self.state.get("oscillators")

        n_car = int(carriers.shape[0])
        n_osc = int(osc.shape[0])
        if n_car == 0 or n_osc == 0:
            return

        P = bonds.get("presence")
        if P.numel() == 0:
            return

        ttl = osc.get("ttl")
        alive = ttl > 0.0
        if not alive.any():
            return

        f_i = osc.get("frequency")
        a_i = osc.get("amplitude")
        phi_i = osc.get("phase")

        # Use bond-energy where present else presence as weights, and gate-high to reduce phase incoherence
        B = bonds.get("energy")
        car_phi = carriers.get("phase")
        car_gate = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)
        dphi = wrap_pi(phi_i.unsqueeze(1) - car_phi.unsqueeze(0))
        g = gate_high(dphi, car_gate.unsqueeze(0))

        W = torch.where(B > 0.0, B, P) * a_i.unsqueeze(1) * g
        W = W * alive.unsqueeze(1).to(DTYPE_REAL)

        sumW = W.sum(dim=0) + self.config.eps  # [M]
        logf = _log_freq(f_i, self.config.eps)  # [N]

        logf_hat = (W * logf.unsqueeze(1)).sum(dim=0) / sumW
        f_hat = torch.exp(logf_hat).to(DTYPE_REAL).clamp(min=1.0)

        # Small inertia: move slower when many atoms are attached
        dof = (P * alive.unsqueeze(1).to(DTYPE_REAL)).sum(dim=0)  # [M]
        alpha = (1.0 / (dof + 5.0)).clamp(0.0, 0.2)  # [M] (no extra knobs)

        f_k = carriers.get("frequency")
        f_k = (1.0 - alpha) * f_k + alpha * f_hat
        carriers.set("frequency", f_k.clamp(min=1.0))
        self.state.set("carriers", carriers)

    # ============================================================
    # Temperature/excitation/gate
    # ============================================================

    def _update_temperature_excitation_gate(self) -> None:
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")

        n_car = int(carriers.shape[0])
        if n_car == 0:
            return

        P = bonds.get("presence")
        dof = P.sum(dim=0) if P.numel() > 0 else torch.zeros(n_car, dtype=DTYPE_REAL, device=self.device)

        heat = carriers.get("heat").clamp(min=0.0)
        T = heat / (dof + 1.0)
        carriers.set("temperature", T)

        # Excitation relaxes toward temperature
        X = carriers.get("excitation")
        dt = float(self.config.dt)
        X = X + dt * (T - X)
        carriers.set("excitation", X)

        # Gate width saturating widening
        base = carriers.get("base_width").clamp(self.config.gate_min, self.config.gate_max)
        w = base * (1.0 + torch.tanh(X))
        w = w.clamp(self.config.gate_min, self.config.gate_max)
        carriers.set("gate_width", w)

        self.state.set("carriers", carriers)

    # ============================================================
    # Costs -> heat
    # ============================================================

    def _drain_bonds_to_heat(self) -> None:
        bonds = self.state.get("bonds")
        carriers = self.state.get("carriers")

        P = bonds.get("presence")
        B = bonds.get("energy")
        if P.numel() == 0 or carriers.shape[0] == 0:
            return

        T = carriers.get("temperature")
        dt = float(self.config.dt)
        scale = float(self.config.hold_cost_scale)

        # Baseline + temperature-scaled drain (baseline enforces "bonds always cost")
        per_bond_cost = (1.0 + T).unsqueeze(0)  # [1,M]
        drain = dt * scale * (P * per_bond_cost)  # [N,M]
        drain = torch.minimum(drain, B)
        B = B - drain

        # Spent bond energy becomes heat
        heat = carriers.get("heat")
        heat = heat + drain.sum(dim=0).to(DTYPE_REAL)
        carriers.set("heat", heat)

        # Snap depleted
        alive = B > 0.0
        P = torch.where(alive, P, torch.zeros_like(P))
        B = torch.where(alive, B, torch.zeros_like(B))

        bonds.set("presence", P)
        bonds.set("energy", B)

        self.state.set("bonds", bonds)
        self.state.set("carriers", carriers)

    def _drain_oscillators_to_heat(self) -> None:
        osc = self.state.get("oscillators")
        bonds = self.state.get("bonds")
        carriers = self.state.get("carriers")

        P = bonds.get("presence")
        if P.numel() == 0 or osc.shape[0] == 0 or carriers.shape[0] == 0:
            return

        ttl = osc.get("ttl")
        alive = ttl > 0.0

        E = osc.get("energy")
        T = carriers.get("temperature")
        dt = float(self.config.dt)
        scale = float(self.config.hold_cost_scale)

        # Temperature of bound carrier (hard assignment) + baseline
        per_osc_cost = (P * (1.0 + T).unsqueeze(0)).sum(dim=1)  # [N]
        per_osc_cost = per_osc_cost * alive.to(DTYPE_REAL)

        drain_i = dt * scale * per_osc_cost
        drain_i = torch.minimum(drain_i, E)
        E = (E - drain_i).clamp(min=0.0)
        osc.set("energy", E)
        self.state.set("oscillators", osc)

        # Deposit to heat per carrier
        heat = carriers.get("heat")
        heat = heat + (P * drain_i.unsqueeze(1)).sum(dim=0).to(DTYPE_REAL)
        carriers.set("heat", heat)
        self.state.set("carriers", carriers)

    # ============================================================
    # Heat diffusion
    # ============================================================

    def _diffuse_heat(self) -> None:
        carriers = self.state.get("carriers")
        n_car = int(carriers.shape[0])
        if n_car <= 1:
            return

        f = carriers.get("frequency")
        w = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)
        U = carriers.get("heat")

        duty = (w / TAU).clamp(min=self.config.eps, max=1.0)
        # adjacency based on relative frequency mismatch under duty
        df = f.unsqueeze(1) - f.unsqueeze(0)
        denom = (f.abs().unsqueeze(1) + f.abs().unsqueeze(0) + self.config.eps)
        rel = df / denom
        scale = (duty.unsqueeze(1) ** 2 + duty.unsqueeze(0) ** 2 + self.config.eps)
        A = torch.exp(-(rel * rel) / scale).to(DTYPE_REAL)

        A.fill_diagonal_(0.0)

        row_sum = A.sum(dim=1) + self.config.eps
        alpha = float(self.config.dt) / float(row_sum.max().item())
        alpha = min(alpha, float(self.config.max_diffusion_alpha))

        dU = alpha * (A @ U - row_sum * U)
        carriers.set("heat", (U + dU).clamp(min=0.0))
        self.state.set("carriers", carriers)

    # ============================================================
    # Spectral profiles (for observers)
    # ============================================================

    def _compute_carrier_spectral_profiles(self, live_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          center_hz: [M]
          var_hz2:   [M]
          is_multimodal: [M] bool (conservative heuristic)
        """
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")
        osc = self.state.get("oscillators")

        m = int(carriers.shape[0])
        if m == 0:
            return (
                torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                torch.empty(0, dtype=DTYPE_REAL, device=self.device),
                torch.empty(0, dtype=torch.bool, device=self.device),
            )

        P = bonds.get("presence")
        if P.numel() == 0 or int(osc.shape[0]) == 0 or live_mask.numel() == 0 or int(live_mask.sum().item()) == 0:
            return (
                torch.zeros(m, dtype=DTYPE_REAL, device=self.device),
                torch.zeros(m, dtype=DTYPE_REAL, device=self.device),
                torch.zeros(m, dtype=torch.bool, device=self.device),
            )

        freq = osc.get("frequency")[live_mask]
        amp = osc.get("amplitude")[live_mask]
        phase = osc.get("phase")[live_mask]
        P_live = P[live_mask]

        # Tuning weights
        tuning = self.tuning_matrix(phase)  # [Nlive,M]
        W = P_live * tuning * amp.unsqueeze(1)  # [Nlive,M]

        sumW = W.sum(dim=0) + self.config.eps  # [M]
        center = (W * freq.unsqueeze(1)).sum(dim=0) / sumW  # [M]
        var = (W * (freq.unsqueeze(1) - center.unsqueeze(0)) ** 2).sum(dim=0) / sumW  # [M]

        # Conservative multimodal heuristic: variance above median and dof>=3
        dof = (P_live > 0.0).sum(dim=0).to(DTYPE_REAL)
        med = var.median()
        is_multi = (var > med) & (dof >= 3.0)

        return center.to(DTYPE_REAL), var.to(DTYPE_REAL), is_multi.to(torch.bool)

    # ============================================================
    # Mitosis (Option A)
    # ============================================================

    def _spectral_mitosis_option_a(self) -> None:
        osc = self.state.get("oscillators")
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")

        P = bonds.get("presence")
        B = bonds.get("energy")

        n_osc = int(osc.shape[0])
        n_car = int(carriers.shape[0])
        if n_car == 0 or n_osc == 0 or P.numel() == 0:
            return
        if n_car >= int(self.config.max_carriers):
            return

        ttl = osc.get("ttl")
        alive_osc = ttl > 0.0

        # Need at least one carrier with >=2 bonded live oscillators
        dof = (P * alive_osc.unsqueeze(1).to(DTYPE_REAL)).sum(dim=0)
        if not (dof >= 2.0).any():
            return

        f_i = osc.get("frequency")
        phi_i = osc.get("phase")
        f_k = carriers.get("frequency")
        phi_k = carriers.get("phase")
        w_k = carriers.get("gate_width").clamp(self.config.gate_min, self.config.gate_max)
        duty = (w_k / TAU).clamp(min=self.config.eps, max=1.0)
        scale_k = duty ** 2 + self.config.eps

        # Distortion measure: relative mismatch scaled by duty
        rel = (f_i.unsqueeze(1) - f_k.unsqueeze(0)) / (f_k.abs().unsqueeze(0) + self.config.eps)  # [N,M]
        mismatch2 = (rel * rel) / scale_k.unsqueeze(0)  # [N,M]

        # Use bond energy where present, else fall back to presence
        W = torch.where(B > 0.0, B, P)
        W = W * alive_osc.unsqueeze(1).to(DTYPE_REAL)

        num = (W * mismatch2).sum(dim=0)
        den = W.sum(dim=0) + self.config.eps
        mismatch_mean = num / den

        coh = carriers.get("coherence").clamp(0.0, 1.0)
        score = mismatch_mean * (1.0 - coh)
        score = torch.where(dof >= 2.0, score, torch.zeros_like(score))

        k_star = int(torch.argmax(score).item())
        if float(score[k_star].item()) <= 0.0:
            return

        bonded_k = (P[:, k_star] > 0.0) & alive_osc
        if not bonded_k.any():
            return

        # Seed: most mismatched bonded oscillator, prefer pulse-high
        dphi_k = wrap_pi(phi_i - phi_k[k_star])
        g_k = gate_high(dphi_k, w_k[k_star]) > 0.0
        mask = bonded_k & g_k
        if not mask.any():
            mask = bonded_k

        mm = mismatch2[:, k_star]
        mm_masked = torch.where(mask, mm, torch.zeros_like(mm))
        i_seed = int(torch.argmax(mm_masked).item())
        if float(mm_masked[i_seed].item()) <= 0.0:
            return

        # New template from real oscillator
        f_new = f_i[i_seed].view(1)
        phi_new = phi_i[i_seed].view(1)

        # Bonded set
        idx = torch.nonzero(bonded_k, as_tuple=False).squeeze(1)
        if idx.numel() < 2:
            return

        f_sub = f_i[idx]

        # Partition by which template reduces distortion
        # Use parent duty/scale for both (new inherits width initially)
        scale = scale_k[k_star]
        rel_old = (f_sub - f_k[k_star]) / (f_k[k_star].abs() + self.config.eps)
        rel_new = (f_sub - f_new) / (f_new.abs() + self.config.eps)

        d_old2 = (rel_old * rel_old) / scale
        d_new2 = (rel_new * rel_new) / scale

        to_new = d_new2 < d_old2
        if (not to_new.any()) or to_new.all():
            return

        if d_old2.sum() <= (d_old2[~to_new].sum() + d_new2[to_new].sum()):
            return

        # --- Perform split ---
        # New carrier identity
        cid = self._next_carrier_id
        self._next_carrier_id += 1
        self._carrier_names.append(f"C{cid}")

        # Partition thermodynamics by moved bond-energy mass (fallback to moved fraction)
        B_col = B[idx, k_star]
        moved_mass = B_col[to_new].sum()
        total_mass = B_col.sum()

        if float(total_mass.item()) > 0.0:
            frac = (moved_mass / (total_mass + self.config.eps)).clamp(0.0, 1.0)
        else:
            frac = to_new.to(DTYPE_REAL).mean().clamp(0.0, 1.0)

        heat = carriers.get("heat")
        excitation = carriers.get("excitation")
        temperature = carriers.get("temperature")

        new_heat = heat[k_star] * frac
        new_exc = excitation[k_star] * frac

        heat[k_star] = heat[k_star] * (1.0 - frac)
        excitation[k_star] = excitation[k_star] * (1.0 - frac)

        base_w_new = carriers.get("base_width")[k_star:k_star + 1].clone()
        gate_w_new = carriers.get("gate_width")[k_star:k_star + 1].clone()

        new_carrier = TensorDict(
            {
                "id": torch.tensor([cid], dtype=torch.int64, device=self.device),
                "frequency": f_new.to(DTYPE_REAL),
                "phase": phi_new.to(DTYPE_REAL),
                "base_width": base_w_new.to(DTYPE_REAL),
                "gate_width": gate_w_new.to(DTYPE_REAL),
                "heat": new_heat.view(1).to(DTYPE_REAL),
                "temperature": torch.zeros(1, dtype=DTYPE_REAL, device=self.device),
                "excitation": new_exc.view(1).to(DTYPE_REAL),
                "coherence": torch.zeros(1, dtype=DTYPE_REAL, device=self.device),
                "energy": torch.zeros(1, dtype=DTYPE_REAL, device=self.device),
            },
            batch_size=[1],
        )

        # Append carriers
        carriers = TensorDict(
            {
                "id": torch.cat([carriers.get("id"), new_carrier.get("id")], dim=0),
                "frequency": torch.cat([carriers.get("frequency"), new_carrier.get("frequency")], dim=0),
                "phase": torch.cat([carriers.get("phase"), new_carrier.get("phase")], dim=0),
                "base_width": torch.cat([carriers.get("base_width"), new_carrier.get("base_width")], dim=0),
                "gate_width": torch.cat([carriers.get("gate_width"), new_carrier.get("gate_width")], dim=0),
                "heat": torch.cat([heat, new_carrier.get("heat")], dim=0),
                "temperature": torch.cat([temperature, new_carrier.get("temperature")], dim=0),
                "excitation": torch.cat([excitation, new_carrier.get("excitation")], dim=0),
                "coherence": torch.cat([carriers.get("coherence"), new_carrier.get("coherence")], dim=0),
                "energy": torch.cat([carriers.get("energy"), new_carrier.get("energy")], dim=0),
            },
            batch_size=[n_car + 1],
        )
        self.state.set("carriers", carriers)

        # Expand bonds
        new_col = n_car
        P = torch.cat([P, torch.zeros(n_osc, 1, dtype=DTYPE_REAL, device=self.device)], dim=1)
        B = torch.cat([B, torch.zeros(n_osc, 1, dtype=DTYPE_REAL, device=self.device)], dim=1)

        idx_move = idx[to_new]
        P[idx_move, new_col] = P[idx_move, k_star]
        B[idx_move, new_col] = B[idx_move, k_star]
        P[idx_move, k_star] = 0.0
        B[idx_move, k_star] = 0.0

        self.state.set("bonds", TensorDict({"presence": P, "energy": B}, batch_size=[]))

        # Extend top-down bias with 0 for new carrier
        if self._top_down_bias_raw.numel() == n_car:
            self._top_down_bias_raw = torch.cat(
                [self._top_down_bias_raw, torch.zeros(1, dtype=DTYPE_REAL, device=self.device)], dim=0
            )
        else:
            self._top_down_bias_raw = torch.zeros(n_car + 1, dtype=DTYPE_REAL, device=self.device)

        if self.config.one_split_per_step:
            return

    # ============================================================
    # Optional: Merge (adjacent by frequency)
    # ============================================================

    def _merge_adjacent_carriers(self) -> None:
        """
        Conservative merge: consider only adjacent carriers in frequency order.

        Merge criterion:
          accept merge if weighted log-frequency distortion doesn't increase.

        Off by default.
        """
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")
        osc = self.state.get("oscillators")

        n_car = int(carriers.shape[0])
        n_osc = int(osc.shape[0])
        if n_car <= 1 or n_osc == 0:
            return

        P = bonds.get("presence")
        if P.numel() == 0:
            return

        ttl = osc.get("ttl")
        alive = ttl > 0.0
        if not alive.any():
            return

        f_k = carriers.get("frequency")
        order = torch.argsort(f_k)

        f_i = osc.get("frequency")
        logf_i = _log_freq(f_i, self.config.eps)
        amp_i = osc.get("amplitude")
        B = bonds.get("energy")

        for s in range(n_car - 1):
            a_idx = int(order[s].item())
            b_idx = int(order[s + 1].item())

            Wa = torch.where(B[:, a_idx] > 0.0, B[:, a_idx], P[:, a_idx]) * amp_i * alive.to(DTYPE_REAL)
            Wb = torch.where(B[:, b_idx] > 0.0, B[:, b_idx], P[:, b_idx]) * amp_i * alive.to(DTYPE_REAL)
            Wunion = Wa + Wb
            if float(Wunion.sum().item()) <= 0.0:
                continue

            logfa = _log_freq(f_k[a_idx], self.config.eps)
            logfb = _log_freq(f_k[b_idx], self.config.eps)

            d2a = (logf_i - logfa) ** 2
            d2b = (logf_i - logfb) ** 2
            cost_sep = (Wa * d2a + Wb * d2b).sum()

            logf_merge = (Wunion * logf_i).sum() / (Wunion.sum() + self.config.eps)
            d2m = (logf_i - logf_merge) ** 2
            cost_merge = (Wunion * d2m).sum()

            if cost_merge <= cost_sep:
                self._merge_two_carriers(keep_idx=a_idx, remove_idx=b_idx, logf_merge=logf_merge)
                if self.config.one_merge_per_step:
                    break

    def _merge_two_carriers(self, *, keep_idx: int, remove_idx: int, logf_merge: torch.Tensor) -> None:
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")

        n_car = int(carriers.shape[0])
        if not (0 <= keep_idx < n_car and 0 <= remove_idx < n_car and keep_idx != remove_idx):
            return

        # Update keep frequency
        f_merge = torch.exp(logf_merge).to(DTYPE_REAL).clamp(min=1.0)
        f = carriers.get("frequency")
        f[keep_idx] = f_merge
        carriers.set("frequency", f)

        # Merge thermodynamics by addition
        for key in ["heat", "excitation", "temperature"]:
            v = carriers.get(key)
            v[keep_idx] = v[keep_idx] + v[remove_idx]
            carriers.set(key, v)

        # Widths: max
        for key in ["base_width", "gate_width"]:
            v = carriers.get(key)
            v[keep_idx] = torch.maximum(v[keep_idx], v[remove_idx])
            carriers.set(key, v)

        # Drop remove carrier row and bond column
        keep_rows = [i for i in range(n_car) if i != remove_idx]
        rows = torch.tensor(keep_rows, dtype=torch.long, device=self.device)

        # Bonds: move remove column into keep then drop remove col
        P = bonds.get("presence")
        B = bonds.get("energy")
        P[:, keep_idx] = torch.clamp(P[:, keep_idx] + P[:, remove_idx], 0.0, 1.0)
        B[:, keep_idx] = B[:, keep_idx] + B[:, remove_idx]
        P = P.index_select(dim=1, index=rows)
        B = B.index_select(dim=1, index=rows)

        self.state.set("bonds", TensorDict({"presence": P, "energy": B}, batch_size=[]))

        carriers_dict = {k: carriers.get(k).index_select(dim=0, index=rows) for k in carriers.keys()}
        self.state.set("carriers", TensorDict(carriers_dict, batch_size=[n_car - 1]))

        # Names + bias alignment
        if 0 <= remove_idx < len(self._carrier_names):
            self._carrier_names.pop(remove_idx)

        if self._top_down_bias_raw.numel() == n_car:
            # average keep+remove, then drop remove
            bias = self._top_down_bias_raw
            bias[keep_idx] = 0.5 * (bias[keep_idx] + bias[remove_idx])
            self._top_down_bias_raw = bias.index_select(dim=0, index=rows)
        else:
            self._top_down_bias_raw = torch.zeros(n_car - 1, dtype=DTYPE_REAL, device=self.device)

    # ============================================================
    # Phase evolution
    # ============================================================

    def _advance_phases(self, dt: float) -> None:
        # Carriers
        carriers = self.state.get("carriers")
        if carriers.shape[0] > 0:
            phase = carriers.get("phase")
            freq = carriers.get("frequency")
            phase = (phase + (TAU * freq) * dt) % TAU
            carriers.set("phase", phase)
            self.state.set("carriers", carriers)

        # Oscillators (only meaningful for live)
        osc = self.state.get("oscillators")
        if osc.shape[0] > 0:
            phase = osc.get("phase")
            freq = osc.get("frequency")
            ttl = osc.get("ttl")
            alive = ttl > 0.0
            phase_new = (phase + (TAU * freq) * dt) % TAU
            phase = torch.where(alive, phase_new, phase)
            osc.set("phase", phase)
            self.state.set("oscillators", osc)

    # ============================================================
    # Synthesis Methods (Top-Down Generation)
    # ============================================================

    def predict_next_token(self, vocab_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Uses carrier 'state_vector' (if present) to predict the next token.

        carriers.state_vector: [N_car, D]
        carriers.energy: [N_car]

        Returns:
            logits: [VocabSize]
            field_vector: [1, D]
        """
        carriers = self.state.get("carriers")
        n_car = int(carriers.shape[0])

        # Check if we have state vectors (custom field for LLM mode)
        if n_car == 0 or "state_vector" not in carriers.keys():
            vocab_size = int(vocab_embeddings.shape[0])
            embed_dim = int(vocab_embeddings.shape[1])
            return (
                torch.zeros(vocab_size, dtype=DTYPE_REAL, device=self.device),
                torch.zeros(1, embed_dim, dtype=DTYPE_REAL, device=self.device),
            )

        vectors = carriers.get("state_vector")               # [N_car, D]
        energy = carriers.get("energy").unsqueeze(1)         # [N_car, 1]

        # The "Ghost Field" is the energy-weighted sum of concepts
        field = (vectors * energy).sum(dim=0, keepdim=True)  # [1, D]

        logits = torch.matmul(field, vocab_embeddings.T).squeeze(0)  # [V]
        return logits.to(DTYPE_REAL), field.to(DTYPE_REAL)

    def diffuse_step(self, dt: float, strength: float = 5.0, noise_scale: float = 10.0, sharpness: float = 1.0) -> None:
        """
        Applies a restoring force to oscillators, pulling them towards carriers.
        
        Uses annealed softmax for smooth transition from soft gravity to hard snapping:
        - Low sharpness (~0.05-0.1): Broad gravity field (softmax). Good for initial distribution.
        - High sharpness (>10.0): Hard snapping (approaches argmax). Good for final locking.
        - Medium sharpness (1.0-5.0): Balanced behavior.
        
        The key insight: Start with low sharpness to sort oscillators into neighborhoods,
        then increase sharpness to lock them precisely. This gives both good distribution
        AND perfect locking.

        Args:
            dt: Time step
            strength: Strength of restoring force (higher = faster convergence)
            noise_scale: Scale of Langevin noise (higher = more exploration)
            sharpness: Controls transition from soft gravity to hard snapping.
                      Low (~0.05) = broad gravity (better distribution)
                      High (>10.0) = hard snap (perfect locking)
                      Default: 1.0 (balanced)

        IMPORTANT FIX vs your previous version:
          - This method now re-bonds *all* oscillators after moving them (rebond_all_oscillators),
            so bond metrics update correctly and `Total bonds` is non-zero.
        """
        osc = self.state.get("oscillators")
        carriers = self.state.get("carriers")

        if osc.shape[0] == 0 or carriers.shape[0] == 0:
            return

        # 1) Distances [N_osc, N_car]
        f_osc = osc.get("frequency").unsqueeze(1)  # [N,1]
        f_car = carriers.get("frequency").unsqueeze(0)  # [1,M]
        dists = f_osc - f_car
        abs_dists = dists.abs()

        # 2) Calculate Target Frequency (Annealed Softmax)
        # We use negative distance because softmax maximizes
        # Scale by sharpness: higher sharpness -> closer to argmax
        # Low sharpness: weights are spread (soft gravity)
        # High sharpness: weights concentrate on nearest (hard snap)
        weights = torch.softmax(-abs_dists * float(sharpness), dim=1)  # [N, M]
        
        # Target is weighted average of carrier frequencies
        target_f = (weights * f_car).sum(dim=1)  # [N]

        # 3) Apply Physics
        current_f = osc.get("frequency")  # [N]
        drift = -float(strength) * (current_f - target_f)
        noise = torch.randn_like(drift) * float(noise_scale)

        new_f = current_f + drift * float(dt) + noise * float(dt)
        new_f = new_f.clamp(min=1.0)

        osc.set("frequency", new_f)
        self.state.set("oscillators", osc)

        # 4) Re-bond (Always use hard assignment for metrics/visuals)
        self.rebond_all_oscillators(conservative=False)

    def _apply_restoring_force(self, dt: float, strength: float = 1.0) -> None:
        """
        Alternative restoring force that uses the current bond matrix (soft field).
        Kept for experimentation; diffuse_step() is the recommended default.
        """
        osc = self.state.get("oscillators")
        carriers = self.state.get("carriers")
        bonds = self.state.get("bonds")

        n_osc = int(osc.shape[0])
        n_car = int(carriers.shape[0])
        if n_osc == 0 or n_car == 0:
            return

        P = bonds.get("presence")  # [N_osc, N_car]
        if P.numel() == 0:
            return

        ttl = osc.get("ttl")
        alive = ttl > 0.0
        if not alive.any():
            return

        target_freqs = carriers.get("frequency")        # [N_car]
        carrier_energies = carriers.get("energy")       # [N_car]

        weights = P * carrier_energies.unsqueeze(0)     # [N_osc, N_car]
        weight_sum = weights.sum(dim=1, keepdim=True) + self.config.eps
        weights_norm = weights / weight_sum

        force_field = (weights_norm * target_freqs.unsqueeze(0)).sum(dim=1)  # [N_osc]

        current_freqs = osc.get("frequency")
        drift = float(strength) * (force_field - current_freqs)

        # Noise proportional to mean temperature
        if n_car > 0:
            T = carriers.get("temperature").mean()
        else:
            T = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)

        noise_scale = torch.sqrt(T.clamp(min=0.0) + self.config.eps)
        noise = torch.randn_like(current_freqs) * noise_scale

        new_freqs = current_freqs + drift * float(dt) + noise * float(dt)
        new_freqs = torch.where(alive, new_freqs, current_freqs).clamp(min=1.0)

        osc.set("frequency", new_freqs)
        self.state.set("oscillators", osc)

    def generate_diffusion(
        self,
        n_oscillators: int,
        carrier_frequencies: Sequence[float],
        n_steps: int = 100,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        cooling_schedule: str = "linear",
    ) -> None:
        """
        Generate signals via thermodynamic diffusion (denoising).

        Backward-compatible signature with your current test harness.

        Fixes:
          - Ensures bonds are created/updated during diffusion by calling rebond_all_oscillators()
            (previously _bond_new_oscillators(new_osc_count=0) was a no-op).
        """
        # Enable synthesis mode
        old_synthesis = self.config.synthesis_mode
        self.config.synthesis_mode = True

        # Initialize carriers with target frequencies
        carrier_frequencies = list(carrier_frequencies)
        n_carriers = len(carrier_frequencies)
        if n_carriers == 0:
            raise ValueError("Must provide at least one carrier frequency")

        carrier_ids = []
        base_w = torch.tensor([0.1], dtype=DTYPE_REAL, device=self.device)

        for _ in carrier_frequencies:
            cid = self._next_carrier_id
            self._next_carrier_id += 1
            self._carrier_names.append(f"C{cid}")
            carrier_ids.append(cid)

        carriers = TensorDict(
            {
                "id": torch.tensor(carrier_ids, dtype=torch.int64, device=self.device),
                "frequency": torch.tensor(carrier_frequencies, dtype=DTYPE_REAL, device=self.device),
                "phase": torch.zeros(n_carriers, dtype=DTYPE_REAL, device=self.device),
                "base_width": base_w.expand(n_carriers).clone(),
                "gate_width": base_w.expand(n_carriers).clone(),
                "heat": torch.full((n_carriers,), float(initial_temperature), dtype=DTYPE_REAL, device=self.device),
                "temperature": torch.full((n_carriers,), float(initial_temperature), dtype=DTYPE_REAL, device=self.device),
                "excitation": torch.zeros(n_carriers, dtype=DTYPE_REAL, device=self.device),
                "coherence": torch.zeros(n_carriers, dtype=DTYPE_REAL, device=self.device),
                "energy": torch.ones(n_carriers, dtype=DTYPE_REAL, device=self.device),
            },
            batch_size=[n_carriers],
        )
        self.state.set("carriers", carriers)

        # Initialize oscillators with random noise
        freq_min, freq_max = 20.0, 20000.0
        random_freqs = torch.rand(n_oscillators, dtype=DTYPE_REAL, device=self.device) * (freq_max - freq_min) + freq_min
        random_phases = torch.rand(n_oscillators, dtype=DTYPE_REAL, device=self.device) * TAU
        random_amps = torch.rand(n_oscillators, dtype=DTYPE_REAL, device=self.device) * 0.5 + 0.1

        osc = TensorDict(
            {
                "frequency": random_freqs,
                "amplitude": random_amps,
                "phase": random_phases,
                "energy": (random_amps * random_freqs.abs()).to(DTYPE_REAL),
                "ttl": torch.full((n_oscillators,), float("inf"), dtype=DTYPE_REAL, device=self.device),
            },
            batch_size=[n_oscillators],
        )
        self.state.set("oscillators", osc)

        bonds = TensorDict(
            {
                "presence": torch.zeros(n_oscillators, n_carriers, dtype=DTYPE_REAL, device=self.device),
                "energy": torch.zeros(n_oscillators, n_carriers, dtype=DTYPE_REAL, device=self.device),
            },
            batch_size=[],
        )
        self.state.set("bonds", bonds)

        self._top_down_bias_raw = torch.zeros(n_carriers, dtype=DTYPE_REAL, device=self.device)

        # Initial bonding so metrics aren't empty
        self.rebond_all_oscillators(conservative=False)

        # Map "temperature" args to a noise schedule (keeps your prior behavior when initial=2.0)
        noise_start = max(float(initial_temperature), 1e-6) * 10.0
        noise_end = max(float(final_temperature), 1e-6) * 10.0

        dt = float(self.config.dt)
        for step in range(int(n_steps)):
            frac = float(step) / float(max(n_steps - 1, 1))

            if cooling_schedule == "linear":
                noise_level = noise_start * (1.0 - frac) + noise_end * frac
                strength = 2.0 + frac * 5.0
                temp = float(initial_temperature) * (1.0 - frac) + float(final_temperature) * frac
            elif cooling_schedule == "exponential":
                alpha = noise_end / max(noise_start, 1e-12)
                noise_level = noise_start * (alpha ** frac)
                strength = 2.0 + (1.0 - alpha ** frac) * 5.0
                temp = float(initial_temperature) * (alpha ** frac)
            else:
                noise_level = 10.0
                strength = 5.0
                temp = float(final_temperature)

            # Update carrier thermodynamics (optional bookkeeping)
            carriers = self.state.get("carriers")
            carriers.set("temperature", torch.full((n_carriers,), temp, dtype=DTYPE_REAL, device=self.device))
            carriers.set("heat", torch.full((n_carriers,), temp, dtype=DTYPE_REAL, device=self.device))
            self.state.set("carriers", carriers)

            # Apply diffusion step (includes rebonding)
            self.diffuse_step(dt=float(dt) * 5.0, strength=float(strength), noise_scale=float(noise_level))

        # Refresh coherence/energy after diffusion
        self._update_carrier_coherence_energy()

        # Restore original synthesis mode
        self.config.synthesis_mode = old_synthesis

    # ============================================================
    # Feedback controllers (internal loop)
    # ============================================================

    def _run_feedback_controllers(self, context: Optional[dict] = None) -> None:
        if not self.feedback_controllers:
            return
        obs = self._observe_base(full_matrices=False)
        for ctrl in self.feedback_controllers:
            try:
                fb = ctrl.feedback(self, obs, context=context)
                if fb:
                    self.provide_feedback(fb)
            except Exception:
                continue

    # ============================================================
    # Metrics
    # ============================================================

    def nnz_P(self) -> int:
        P = self.state.get("bonds").get("presence")
        if P.numel() == 0:
            return 0
        return int((P > 0.0).sum().item())

    def global_sync_R(self) -> float:
        osc = self.state.get("oscillators")
        if osc.shape[0] == 0:
            return 0.0
        phases = osc.get("phase")
        amps = osc.get("amplitude")
        ttl = osc.get("ttl")
        active = (amps > self.config.eps) & (ttl > 0.0)
        if not active.any():
            return 0.0
        phasors = torch.exp(1j * phases[active].to(torch.complex64))
        R = torch.abs(phasors.mean())
        return float(R.item())

    def L_comp(self) -> int:
        return self.nnz_P() + int(self.state.get("carriers").shape[0])


# ============================================================
# Example internal feedback controller (optional)
# ============================================================

class CoherenceAttractorController:
    """
    Simple internal loop (example):
      - Carriers with above-mean coherence get positive utility
      - Below-mean coherence get negative utility
    """
    def __init__(self, strength: float = 1.0):
        self.strength = float(strength)

    def feedback(self, manifold: Manifold, obs: dict, context: Optional[dict] = None) -> Optional[dict]:
        m = int(obs.get("n_carriers", 0))
        if m == 0:
            return None
        coh: torch.Tensor = obs["carrier_coherence"]
        u = coh - coh.mean()
        u = u * self.strength
        ids = obs["carrier_ids"]
        return {"carrier_utility": {int(ids[i]): float(u[i].item()) for i in range(m)}, "utility": 1.0}


# ============================================================
# Audio separation utilities (end-to-end test harness)
# ============================================================

def _load_audio_any(path: str) -> tuple[torch.Tensor, int]:
    """
    Returns:
      audio: [C, T] float32 in [-1, 1] (best effort)
      sr: int
    Uses soundfile if available, else scipy.io.wavfile.
    """
    try:
        import soundfile as sf  # type: ignore
        x, sr = sf.read(path, always_2d=True)
        # x: [T, C]
        audio = torch.from_numpy(x.T).to(dtype=torch.float32)
        return audio, int(sr)
    except Exception:
        from scipy.io import wavfile  # type: ignore
        sr, x = wavfile.read(path)
        if x.ndim == 1:
            x = x[:, None]
        # x: [T, C]
        # Normalize integers
        if x.dtype.kind in ("i", "u"):
            maxv = float((1 << (8 * x.dtype.itemsize - 1)) - 1)
            x_f = x.astype("float32") / maxv
        else:
            x_f = x.astype("float32")
        audio = torch.from_numpy(x_f.T)
        return audio, int(sr)


def _save_audio_any(path: str, audio: torch.Tensor, sr: int) -> None:
    """
    audio: [C, T] float32
    Writes with soundfile if available, else scipy.io.wavfile (16-bit PCM).
    """
    audio = audio.detach().cpu().to(torch.float32)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    x = audio.T.numpy()  # [T,C]

    try:
        import soundfile as sf  # type: ignore
        sf.write(path, x, sr)
        return
    except Exception:
        from scipy.io import wavfile  # type: ignore
        x16 = (x.clip(-1.0, 1.0) * 32767.0).astype("int16")
        wavfile.write(path, sr, x16)


def _bin_frequencies(sr: int, n_fft: int, device: torch.device) -> torch.Tensor:
    # STFT uses bins k=0..n_fft//2
    f = torch.arange(0, n_fft // 2 + 1, device=device, dtype=torch.float32) * (float(sr) / float(n_fft))
    return f.to(DTYPE_REAL)


def _select_bins_by_energy(mag: torch.Tensor, energy_keep: float = 0.99, max_bins: int = 128) -> torch.Tensor:
    """
    Pick the smallest set of bins that explain ~energy_keep of frame energy (capped by max_bins).
    """
    power = mag * mag
    tot = power.sum()
    if float(tot.item()) <= 0.0:
        return torch.empty(0, dtype=torch.long, device=mag.device)

    sorted_pow, idx = torch.sort(power, descending=True)
    cdf = torch.cumsum(sorted_pow, dim=0) / (tot + 1e-12)
    k = int((cdf <= float(energy_keep)).sum().item()) + 1
    k = max(1, min(k, int(max_bins), int(mag.numel())))
    return idx[:k]


def separate_audio_two_speakers_zero_shot(
    wav_path: str,
    out_dir: str,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    energy_keep: float = 0.99,
    max_bins_per_frame: int = 128,
    target_outputs: int = 2,
    device: Optional[torch.device] = None,
) -> list[str]:
    """
    End-to-end test harness:

    1) Load wav (mono or stereo)
    2) STFT
    3) For each frame:
         - extract sparse "signals" from bins (freq/amp/phase, duration=win_len/sr)
         - manifold.step(signals)
         - compute per-carrier soft masks for all bins (tf_mask)
         - store masks keyed by carrier id
    4) Pick top carriers by explained STFT energy
    5) Apply masks to original STFT and iSTFT to audio files
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or torch.device("cpu")
    win_length = int(win_length or n_fft)

    audio, sr = _load_audio_any(wav_path)
    audio = audio.to(device=device, dtype=torch.float32)
    C, T = int(audio.shape[0]), int(audio.shape[1])

    window = torch.hann_window(win_length, device=device, dtype=torch.float32)

    # STFT per channel: [C, F, Frames]
    Xc = []
    for c in range(C):
        X = torch.stft(
            audio[c],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        Xc.append(X)
    Xc_t = torch.stack(Xc, dim=0)  # [C,F,Frames]
    F = int(Xc_t.shape[1])
    Frames = int(Xc_t.shape[2])

    # Mono mixture used for mask estimation
    X_mono = Xc_t.mean(dim=0)  # [F,Frames]

    # Build manifold with dt matched to hop duration
    cfg = ManifoldConfig(
        dt=float(hop_length) / float(sr),
        hold_cost_scale=1.0,
        internal_feedback_enabled=False,
    )
    m = Manifold(cfg, device=device)

    bin_freqs = _bin_frequencies(sr, n_fft, device=device)  # [F]

    # Store masks per frame keyed by carrier id
    frame_masks: list[dict[int, torch.Tensor]] = []

    # Duration per oscillator = window length in seconds
    atom_duration = torch.full((1,), float(win_length) / float(sr), dtype=DTYPE_REAL, device=device)

    for t in range(Frames):
        spec = X_mono[:, t]              # [F] complex
        mag = spec.abs().to(DTYPE_REAL)  # [F]
        ph = torch.angle(spec).to(DTYPE_REAL)

        idx = _select_bins_by_energy(mag, energy_keep=energy_keep, max_bins=max_bins_per_frame)
        if idx.numel() > 0:
            signals_td = TensorDict(
                {
                    "frequency": bin_freqs[idx],
                    "amplitude": mag[idx],
                    "phase": ph[idx],
                    "duration": atom_duration.expand(idx.shape[0]).clone(),
                },
                batch_size=[int(idx.numel())],
            )
        else:
            signals_td = TensorDict(
                {
                    "frequency": torch.empty(0, dtype=DTYPE_REAL, device=device),
                    "amplitude": torch.empty(0, dtype=DTYPE_REAL, device=device),
                    "phase": torch.empty(0, dtype=DTYPE_REAL, device=device),
                    "duration": torch.empty(0, dtype=DTYPE_REAL, device=device),
                },
                batch_size=[0],
            )

        m.step(signals_td)

        # Compute masks for *all* bins this frame using current carriers
        mask_ft = m.tf_mask(bin_freqs, ph)  # [F, M_current]
        carriers = m.state.get("carriers")
        ids = carriers.get("id").tolist() if carriers.shape[0] > 0 else []

        d: dict[int, torch.Tensor] = {}
        for j, cid in enumerate(ids):
            d[int(cid)] = mask_ft[:, j].detach().clone()
        frame_masks.append(d)

    # Assemble per-carrier mask matrices [F,Frames] for all seen carrier ids
    all_ids: set[int] = set()
    for d in frame_masks:
        all_ids.update(d.keys())
    all_ids_sorted = sorted(all_ids)

    # Precompute mixture power for scoring
    power = (X_mono.abs() ** 2).to(DTYPE_REAL)  # [F,Frames]

    masks_by_id: dict[int, torch.Tensor] = {}
    scores: dict[int, float] = {}

    for cid in all_ids_sorted:
        Mft = torch.zeros(F, Frames, dtype=DTYPE_REAL, device=device)
        for t in range(Frames):
            v = frame_masks[t].get(cid, None)
            if v is not None:
                Mft[:, t] = v
        masks_by_id[cid] = Mft
        scores[cid] = float((Mft * power).sum().item())

    # Pick top carriers by explained energy
    ranked = sorted(all_ids_sorted, key=lambda k: scores.get(k, 0.0), reverse=True)
    chosen = ranked[: max(1, int(target_outputs))]

    out_paths: list[str] = []

    # Reconstruct each chosen source
    for rank, cid in enumerate(chosen):
        Mft = masks_by_id[cid]  # [F,Frames]
        Yc = Xc_t * Mft.unsqueeze(0)  # [C,F,Frames] complex (broadcast)

        # iSTFT per channel
        y_ch = []
        for c in range(C):
            y = torch.istft(
                Yc[c],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                length=T,
            )
            y_ch.append(y)
        y_out = torch.stack(y_ch, dim=0)  # [C,T]

        out_path = os.path.join(out_dir, f"source_{rank:02d}_carrier_{cid}.wav")
        _save_audio_any(out_path, y_out, sr)
        out_paths.append(out_path)

    return out_paths


# ============================================================
# Minimal example
# ============================================================

if __name__ == "__main__":
    # Example:
    #   python manifold.py /path/to/two_speakers.wav out_sep/
    import sys

    if len(sys.argv) >= 3:
        inp = sys.argv[1]
        outd = sys.argv[2]
        paths = separate_audio_two_speakers_zero_shot(inp, outd)
        print("Wrote:")
        for p in paths:
            print("  ", p)
    else:
        print("Usage: python manifold.py <mix.wav> <out_dir>")
