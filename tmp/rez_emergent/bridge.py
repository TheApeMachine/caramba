
"""
rez_emergent.bridge

A domain bridge implemented as another manifold (not a hard-coded lookup).

The original demo used a direct word→audio dictionary mapping. 
That breaks the internal thermodynamic story because the "translation" is not an
energy-flow process.

Here we implement:
- a BridgeManifold that learns a semantic→spectral association via local co-activation,
  using the same carrier-field mechanism as the semantic manifold.

No backprop. No gradient descent. No tuned learning rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import math
import torch

from physics import DTYPE_REAL, PhysicsConfig
from indexing import LSHIndex


def _safe_std(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.numel() <= 1:
        return x.abs().mean().clamp_min(eps)
    return x.std(unbiased=False).clamp_min(eps)


def _normalize_vec(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm().clamp_min(eps)


def _normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


@dataclass
class BridgeMetrics:
    carriers: int
    nucleation_mass: float
    carrier_energy_mean: float
    carrier_heat_mean: float
    last_debug: dict[str, Any]


class CarrierBridge:
    """
    Generic carrier field mapping:
        in_vec -> out_vec

    Carriers store:
        in_pos  (prototype in input space)
        out_pos (prototype in output space)
        energy, heat, age

    Update is local and scale-derived.
    """

    def __init__(self, config: PhysicsConfig, device: torch.device, in_dim: int, out_dim: int):
        self.config = config
        self.device = device
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.carriers: dict[str, torch.Tensor] = {
            "in_pos": torch.empty((0, self.in_dim), dtype=DTYPE_REAL, device=device),
            "out_pos": torch.empty((0, self.out_dim), dtype=DTYPE_REAL, device=device),
            "energy": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "heat": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "age": torch.empty((0,), dtype=DTYPE_REAL, device=device),
        }
        self.nucleation_mass: float = 0.0
        self.last_debug: dict[str, Any] = {}

    def _carrier_count(self) -> int:
        return int(self.carriers["energy"].shape[0])

    def _append_carrier(self, in_pos: torch.Tensor, out_pos: torch.Tensor, energy: torch.Tensor) -> None:
        eps = float(self.config.eps)
        in_pos = _normalize_rows(in_pos.to(self.device, dtype=DTYPE_REAL).view(1, -1), eps)
        out_pos = _normalize_rows(out_pos.to(self.device, dtype=DTYPE_REAL).view(1, -1), eps)
        energy = energy.to(self.device, dtype=DTYPE_REAL).view(1)
        heat = torch.zeros((1,), dtype=DTYPE_REAL, device=self.device)
        age = torch.zeros((1,), dtype=DTYPE_REAL, device=self.device)

        if self._carrier_count() == 0:
            self.carriers = {"in_pos": in_pos, "out_pos": out_pos, "energy": energy, "heat": heat, "age": age}
            return
        self.carriers["in_pos"] = torch.cat([self.carriers["in_pos"], in_pos], dim=0)
        self.carriers["out_pos"] = torch.cat([self.carriers["out_pos"], out_pos], dim=0)
        self.carriers["energy"] = torch.cat([self.carriers["energy"], energy], dim=0)
        self.carriers["heat"] = torch.cat([self.carriers["heat"], heat], dim=0)
        self.carriers["age"] = torch.cat([self.carriers["age"], age], dim=0)

    def _weights(self, in_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = float(self.config.eps)
        M = self._carrier_count()
        if M == 0:
            return (
                torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device),
            )
        in_pos = self.carriers["in_pos"]  # [M, in_dim]
        in_vec = _normalize_vec(in_vec.to(self.device, dtype=DTYPE_REAL), eps)
        sim = torch.mv(in_pos, in_vec)
        d = (1.0 - sim).clamp_min(0.0)
        d_scale = d.min().clamp_min(eps)

        heat = self.carriers["heat"].clamp_min(0.0)
        h_scale = heat.mean().clamp_min(eps)
        temp_factor = (h_scale + heat) / h_scale
        T_eff = (d_scale * temp_factor).clamp_min(eps)

        logits = -d / T_eff
        logits = logits - logits.max()
        w = torch.exp(logits)
        w = w / w.sum().clamp_min(eps)
        return w, d, d_scale

    def _prune_dead(self) -> None:
        M = self._carrier_count()
        if M == 0:
            return
        eps = float(self.config.eps)
        E = self.carriers["energy"]
        E_scale = E.abs().mean().clamp_min(eps)
        alive = E > (eps * E_scale)
        if alive.all():
            return
        if alive.any():
            for k in list(self.carriers.keys()):
                self.carriers[k] = self.carriers[k][alive]
        else:
            self.carriers = {
                "in_pos": torch.empty((0, self.in_dim), dtype=DTYPE_REAL, device=self.device),
                "out_pos": torch.empty((0, self.out_dim), dtype=DTYPE_REAL, device=self.device),
                "energy": torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                "heat": torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                "age": torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
            }

    def observe(self, in_vec: torch.Tensor, out_vec: torch.Tensor, mass: Optional[torch.Tensor] = None) -> None:
        dt = float(self.config.dt)
        eps = float(self.config.eps)

        in_vec = _normalize_vec(in_vec.to(self.device, dtype=DTYPE_REAL), eps)
        out_vec = _normalize_vec(out_vec.to(self.device, dtype=DTYPE_REAL), eps)
        if mass is None:
            mass_t = torch.tensor(1.0, dtype=DTYPE_REAL, device=self.device)
        else:
            mass_t = mass.to(self.device, dtype=DTYPE_REAL).clamp_min(eps)

        if self._carrier_count() == 0:
            self._append_carrier(in_vec, out_vec, energy=mass_t.detach())
            self.last_debug = {"mass": float(mass_t.item()), "carriers": 1, "nucleated": 1}
            return

        w, d, d_scale = self._weights(in_vec)
        intake = w * mass_t
        mean_intake = intake.mean().clamp_min(eps)

        # Prediction mismatch (pre-update)
        pred_pre = torch.mv(self.carriers["out_pos"].t(), w)
        pred_pre = _normalize_vec(pred_pre, eps)
        sim_pred = (pred_pre * out_vec).sum().clamp(-1.0, 1.0)
        mismatch = (0.5 * (1.0 - sim_pred)).clamp(0.0, 1.0)

        # Metabolism
        E = self.carriers["energy"]
        E_scale = E.mean().abs().clamp_min(eps)
        cost_rate = (E / E_scale).clamp_min(0.0)
        E = (E * torch.exp(-dt * cost_rate) + dt * intake).clamp_min(0.0)
        self.carriers["energy"] = E

        # Drift
        age = self.carriers["age"]
        inertia = (E * (age + dt)).clamp_min(0.0)
        rate = (intake / (intake + inertia + eps)).clamp(0.0, 1.0)

        in_pos = self.carriers["in_pos"]
        out_pos = self.carriers["out_pos"]
        in_pos = in_pos + (dt * rate).unsqueeze(1) * (in_vec.unsqueeze(0) - in_pos)
        out_pos = out_pos + (dt * rate).unsqueeze(1) * (out_vec.unsqueeze(0) - out_pos)
        self.carriers["in_pos"] = _normalize_rows(in_pos, eps)
        self.carriers["out_pos"] = _normalize_rows(out_pos, eps)

        # Heat
        heat = self.carriers["heat"]
        heat = heat + dt * (intake * (d / d_scale.clamp_min(eps)))
        h_scale = heat.mean().clamp_min(eps)
        heat = heat * torch.exp(torch.tensor(-dt, device=heat.device, dtype=heat.dtype) / h_scale)
        self.carriers["heat"] = heat

        # Age
        self.carriers["age"] = self.carriers["age"] + dt

        # Nucleation pressure from mismatch (computed pre-update)
        self.nucleation_mass += float((dt * mismatch).item())
        nucleated = 0
        while self.nucleation_mass >= 1.0:
            self.nucleation_mass -= 1.0
            self._append_carrier(in_vec, out_vec, energy=mean_intake.detach())
            nucleated += 1

        self._prune_dead()

        self.last_debug = {
            "mass": float(mass_t.item()),
            "carriers": self._carrier_count(),
            "mean_intake": float(mean_intake.item()),
            "mismatch": float(mismatch.item()),
            "nucleated": nucleated,
            "w_max": float(w.max().item()),
        }

    def predict(self, in_vec: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.eps)
        in_vec = _normalize_vec(in_vec.to(self.device, dtype=DTYPE_REAL), eps)
        if self._carrier_count() == 0:
            return torch.zeros((self.out_dim,), dtype=DTYPE_REAL, device=self.device)
        w, _, _ = self._weights(in_vec)
        out_pos = self.carriers["out_pos"]
        pred = torch.mv(out_pos.t(), w)
        return _normalize_vec(pred, eps)

    def metrics(self) -> BridgeMetrics:
        M = self._carrier_count()
        if M == 0:
            return BridgeMetrics(carriers=0, nucleation_mass=float(self.nucleation_mass), carrier_energy_mean=0.0, carrier_heat_mean=0.0, last_debug=dict(self.last_debug))
        e = self.carriers["energy"]
        h = self.carriers["heat"]
        return BridgeMetrics(
            carriers=M,
            nucleation_mass=float(self.nucleation_mass),
            carrier_energy_mean=float(e.mean().item()),
            carrier_heat_mean=float(h.mean().item()),
            last_debug=dict(self.last_debug),
        )


class BridgeManifold:
    """
    Semantic -> Spectral transduction.

    - Spectral bank: fixed frequency bins with fixed embeddings.
    - Bridge carriers learn to map semantic vectors to spectral embeddings
      via co-activation experience.

    Training usage:
      bridge.observe(sem_vec, audio_vec)

    Inference usage:
      freqs, energies = bridge.decode_targets(bridge.predict(sem_vec))
    """

    def __init__(
        self,
        config: PhysicsConfig,
        device: torch.device,
        sem_dim: int,
        spec_bins: int,
        spec_min_hz: float,
        spec_max_hz: float,
        spec_embed_dim: Optional[int] = None,
        enable_event_horizon: bool = True,
        lsh_seed: Optional[int] = None,
    ):
        self.config = config
        self.device = device
        self.sem_dim = int(sem_dim)
        self.spec_bins = int(spec_bins)
        self.spec_min_hz = float(spec_min_hz)
        self.spec_max_hz = float(spec_max_hz)

        eps = float(self.config.eps)

        if self.spec_bins <= 0:
            raise ValueError("spec_bins must be > 0")
        if self.spec_min_hz <= 0 or self.spec_max_hz <= 0 or not (self.spec_min_hz < self.spec_max_hz):
            raise ValueError("spec_min_hz and spec_max_hz must be positive and spec_min_hz < spec_max_hz")

        # Spectral embedding dimension (derived if not provided)
        if spec_embed_dim is None:
            # Derive from bins (no fixed magic): ceil(sqrt(B)) but at least 1
            spec_embed_dim = int(math.ceil(math.sqrt(float(self.spec_bins))))
        self.spec_embed_dim = int(spec_embed_dim)

        # Fixed spectral bank
        # Log-spaced bins resemble auditory spacing.
        log_f = torch.linspace(math.log(self.spec_min_hz), math.log(self.spec_max_hz), self.spec_bins, device=device, dtype=DTYPE_REAL)
        self.spec_freqs = torch.exp(log_f)

        # Fixed embeddings for bins
        emb = torch.randn((self.spec_bins, self.spec_embed_dim), device=device, dtype=DTYPE_REAL)
        emb = _normalize_rows(emb, eps)
        self.spec_emb = emb

        # Optional index for decoding when bins are large
        if enable_event_horizon and (self.spec_bins > (self.spec_embed_dim * self.spec_embed_dim)):
            self.spec_index: Optional[LSHIndex] = LSHIndex(self.spec_emb, eps=eps, seed=lsh_seed)
        else:
            self.spec_index = None

        # Carrier bridge
        self.bridge = CarrierBridge(config=config, device=device, in_dim=self.sem_dim, out_dim=self.spec_embed_dim)

    def encode_freqs(self, freqs: torch.Tensor, energies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert a set of active frequencies into a spectral embedding vector
        by soft-assigning them to the fixed bank.
        """
        eps = float(self.config.eps)

        freqs = freqs.to(self.device, dtype=DTYPE_REAL).flatten()
        if freqs.numel() == 0:
            return torch.zeros((self.spec_embed_dim,), dtype=DTYPE_REAL, device=self.device)

        if energies is None:
            e = torch.ones_like(freqs)
        else:
            e = energies.to(self.device, dtype=DTYPE_REAL).flatten()
            if e.shape != freqs.shape:
                raise ValueError("energies must match freqs shape")

        # Assign each freq to nearest bin (1D)
        # Distances in log space.
        log_f = torch.log(freqs.clamp_min(eps))
        log_bins = torch.log(self.spec_freqs.clamp_min(eps))
        # [n, B]
        d = torch.abs(log_f.unsqueeze(1) - log_bins.unsqueeze(0))
        d_scale = torch.sqrt(torch.mean(d * d)).clamp_min(eps)
        w = torch.softmax(-d / d_scale, dim=1)  # [n, B]

        mass = e.sum().clamp_min(eps)
        w = w * (e / mass).unsqueeze(1)
        bin_mass = w.sum(dim=0)  # [B]
        vec = torch.mv(self.spec_emb.t(), bin_mass)
        return _normalize_vec(vec, eps)

    def observe(self, sem_vec: torch.Tensor, audio_freqs: torch.Tensor, audio_energies: Optional[torch.Tensor] = None) -> None:
        """
        Learn from co-activation: semantic vector alongside an observed audio pattern.
        """
        eps = float(self.config.eps)
        sem_vec = sem_vec.to(self.device, dtype=DTYPE_REAL)
        sem_vec = _normalize_vec(sem_vec, eps)
        out_vec = self.encode_freqs(audio_freqs, audio_energies)
        # Mass derived from audio energy scale
        if audio_energies is None:
            mass = torch.tensor(1.0, dtype=DTYPE_REAL, device=self.device)
        else:
            mass = audio_energies.to(self.device, dtype=DTYPE_REAL).sum().clamp_min(eps)
        self.bridge.observe(sem_vec, out_vec, mass=mass)

    def predict_vec(self, sem_vec: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.eps)
        sem_vec = _normalize_vec(sem_vec.to(self.device, dtype=DTYPE_REAL), eps)
        return self.bridge.predict(sem_vec)

    def decode_targets(self, spec_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a spectral embedding into (freqs, energies).

        Energies are a normalized distribution over selected bins.
        """
        eps = float(self.config.eps)
        spec_vec = _normalize_vec(spec_vec.to(self.device, dtype=DTYPE_REAL), eps)

        B = self.spec_bins
        k = int(math.ceil(math.sqrt(float(B))))
        if self.spec_index is None:
            sims = torch.mv(self.spec_emb, spec_vec)  # [B]
            # Select top-k bins
            vals, idx = torch.topk(sims, k=min(k, B), largest=True, sorted=True)
        else:
            idx, vals = self.spec_index.topk(spec_vec, k=k)

        if idx.numel() == 0:
            return torch.empty((0,), dtype=DTYPE_REAL, device=self.device), torch.empty((0,), dtype=DTYPE_REAL, device=self.device)

        s_scale = _safe_std(vals, eps)
        logits = vals / s_scale
        probs = torch.softmax(logits, dim=0)

        freqs = self.spec_freqs.index_select(0, idx)
        energies = probs  # already sums to 1
        return freqs, energies

    def metrics(self) -> BridgeMetrics:
        return self.bridge.metrics()