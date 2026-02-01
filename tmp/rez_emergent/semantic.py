
"""
rez_emergent.semantic

Semantic manifold as a self-organizing carrier field (no dense V×V grammar matrix).

This module replaces the earlier monolithic "step_grammar" approach which mixed:
- context decay
- topology updates
- metabolic ledgers
- flow propagation
in one large function. 

Key properties:
- No backpropagation.
- No trainable hyperparameters (rates are derived from instantaneous scales).
- Scales by:
  - Keeping the carrier population dynamic (not V×V).
  - Optional "semantic event horizon" for the output projection using LSH
    (avoid full V dot-products when V is very large).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import math
import torch

from physics import DTYPE_REAL, OutputState, PhysicsConfig
from indexing import LSHIndex


def _safe_std(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.numel() <= 1:
        return x.abs().mean().clamp_min(eps)
    return x.std(unbiased=False).clamp_min(eps)


def _normalize_vec(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm().clamp_min(eps)


def _normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def _n_rows(state: dict[str, torch.Tensor]) -> int:
    pos = state.get("position")
    if not isinstance(pos, torch.Tensor) or pos.dim() == 0:
        return 0
    return int(pos.shape[0])


@dataclass
class SemanticMetrics:
    carriers: int
    nucleation_mass: float
    carrier_energy_mean: float
    carrier_heat_mean: float
    last_debug: dict[str, Any]


class SemanticManifold:
    """
    Streaming semantic manifold for next-token prediction.

    State:
      - token attractors (fixed geometry): position ∈ R^{V×D}
      - context particles: sequence embeddings + emergent recency energy
      - carriers: dynamic low-rank context→next mapping

    Learning:
      observe_next(next_token_id): local energetic adaptation of carriers
      (no gradients; no explicit loss minimization)
    """

    def __init__(
        self,
        config: PhysicsConfig,
        device: torch.device,
        embed_dim: int,
        vocab_size: int,
        token_embeddings: Optional[torch.Tensor] = None,
        enable_event_horizon: bool = True,
        lsh_seed: Optional[int] = None,
    ):
        self.config = config
        self.device = device
        self.embed_dim = int(embed_dim)
        self.vocab_size = int(vocab_size)
        self.enable_event_horizon = bool(enable_event_horizon)

        eps = float(self.config.eps)

        # --- Token geometry (fixed) ---
        if token_embeddings is None:
            if embed_dim >= vocab_size:
                emb = torch.eye(vocab_size, embed_dim, dtype=DTYPE_REAL, device=device)
            else:
                emb = torch.randn(vocab_size, embed_dim, dtype=DTYPE_REAL, device=device)
            emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(eps)
        else:
            emb = token_embeddings.to(device=device, dtype=DTYPE_REAL)
            if emb.shape != (vocab_size, embed_dim):
                raise ValueError(f"token_embeddings must be {(vocab_size, embed_dim)}, got {tuple(emb.shape)}")
            emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(eps)

        self.attractors: dict[str, torch.Tensor] = {
            "id": torch.arange(vocab_size, dtype=torch.int64, device=device),
            "position": emb,
        }

        # --- Event horizon index (optional) ---
        # Use index when V is large enough that full projection is expensive.
        # Criterion is derived (no fixed numeric cutoff): use LSH when V > D^2.
        if self.enable_event_horizon and (self.vocab_size > (self.embed_dim * self.embed_dim)):
            self.vocab_index: Optional[LSHIndex] = LSHIndex(self.attractors["position"], eps=eps, seed=lsh_seed)
        else:
            self.vocab_index = None

        # --- Context particles (set by ingest_context) ---
        self.particles: dict[str, torch.Tensor] = {}

        # --- Dynamic carriers ---
        self.carriers: dict[str, torch.Tensor] = {
            "ctx_pos": torch.empty((0, embed_dim), dtype=DTYPE_REAL, device=device),
            "out_pos": torch.empty((0, embed_dim), dtype=DTYPE_REAL, device=device),
            "energy": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "heat": torch.empty((0,), dtype=DTYPE_REAL, device=device),
            "age": torch.empty((0,), dtype=DTYPE_REAL, device=device),
        }
        self.nucleation_mass: float = 0.0
        self.last_debug: dict[str, Any] = {}

    # ----------------------------
    # Context ingestion
    # ----------------------------

    def ingest_context(self, embeddings: torch.Tensor) -> None:
        embeddings = embeddings.to(self.device, dtype=DTYPE_REAL)
        if embeddings.dim() != 2 or embeddings.shape[1] != self.embed_dim:
            raise ValueError(f"embeddings must be [N,{self.embed_dim}], got {tuple(embeddings.shape)}")
        n = int(embeddings.shape[0])
        if n == 0:
            self.particles = {}
            return

        # Recency energy emerges from order.
        idx = torch.arange(1, n + 1, device=self.device, dtype=DTYPE_REAL)
        energy = idx / idx.max().clamp_min(self.config.eps)
        self.particles = {"position": embeddings, "energy": energy}

    def _context_vector_and_mass(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = float(self.config.eps)
        if _n_rows(self.particles) == 0:
            ctx = torch.zeros((self.embed_dim,), dtype=DTYPE_REAL, device=self.device)
            mass = torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
            return ctx, mass
        p = self.particles["position"]
        e = self.particles["energy"]
        mass = e.sum().clamp_min(eps)
        w = e / mass
        ctx = (w.unsqueeze(1) * p).sum(dim=0)
        ctx = _normalize_vec(ctx, eps)
        return ctx, mass

    # ----------------------------
    # Carrier mechanics
    # ----------------------------

    def _carrier_count(self) -> int:
        return int(self.carriers["energy"].shape[0])

    def _append_carrier(self, ctx_pos: torch.Tensor, out_pos: torch.Tensor, energy: torch.Tensor) -> None:
        eps = float(self.config.eps)

        ctx_pos = _normalize_rows(ctx_pos.to(self.device, dtype=DTYPE_REAL).view(1, -1), eps)
        out_pos = _normalize_rows(out_pos.to(self.device, dtype=DTYPE_REAL).view(1, -1), eps)
        energy = energy.to(self.device, dtype=DTYPE_REAL).view(1)
        heat = torch.zeros((1,), dtype=DTYPE_REAL, device=self.device)
        age = torch.zeros((1,), dtype=DTYPE_REAL, device=self.device)

        if self._carrier_count() == 0:
            self.carriers = {
                "ctx_pos": ctx_pos,
                "out_pos": out_pos,
                "energy": energy,
                "heat": heat,
                "age": age,
            }
            return

        self.carriers["ctx_pos"] = torch.cat([self.carriers["ctx_pos"], ctx_pos], dim=0)
        self.carriers["out_pos"] = torch.cat([self.carriers["out_pos"], out_pos], dim=0)
        self.carriers["energy"] = torch.cat([self.carriers["energy"], energy], dim=0)
        self.carriers["heat"] = torch.cat([self.carriers["heat"], heat], dim=0)
        self.carriers["age"] = torch.cat([self.carriers["age"], age], dim=0)

    def _carrier_weights(self, ctx_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          w: [M] distribution over carriers
          d: [M] distances (1 - cosine similarity)
          d_scale: scalar (local temperature scale)
        """
        eps = float(self.config.eps)
        M = self._carrier_count()
        if M == 0:
            return (
                torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device),
            )

        ctx_pos = self.carriers["ctx_pos"]  # [M,D]
        sim = torch.mv(ctx_pos, ctx_vec)  # [M]
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

    def _prune_dead_carriers(self) -> None:
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
                "ctx_pos": torch.empty((0, self.embed_dim), dtype=DTYPE_REAL, device=self.device),
                "out_pos": torch.empty((0, self.embed_dim), dtype=DTYPE_REAL, device=self.device),
                "energy": torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                "heat": torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
                "age": torch.empty((0,), dtype=DTYPE_REAL, device=self.device),
            }

    # ----------------------------
    # Learning: observe transitions
    # ----------------------------

    def observe_next(self, next_token_id: int) -> None:
        """
        Update carriers using an observed next token.

        This is not gradient descent; it's local adaptation driven by energetic intake.
        """
        dt = float(self.config.dt)
        eps = float(self.config.eps)

        ctx_vec, ctx_mass = self._context_vector_and_mass()
        next_vec = self.attractors["position"][int(next_token_id)]

        if float(ctx_mass.item()) == 0.0:
            self.last_debug = {"ctx_mass": 0.0, "carriers": self._carrier_count()}
            return

        if self._carrier_count() == 0:
            self._append_carrier(ctx_vec, next_vec, energy=ctx_mass.detach())
            self.last_debug = {"ctx_mass": float(ctx_mass.item()), "carriers": 1, "nucleated": 1}
            return

        # ---- Phase A: topology (resonance weights) ----
        w, d, d_scale = self._carrier_weights(ctx_vec)
        intake = w * ctx_mass
        mean_intake = intake.mean().clamp_min(eps)

        # ---- Phase B: prediction mismatch (pre-update) ----
        # Compute mismatch BEFORE adapting prototypes; this drives true emergence.
        pred_vec_pre = torch.mv(self.carriers["out_pos"].t(), w)
        pred_vec_pre = _normalize_vec(pred_vec_pre, eps)
        sim_pred = (pred_vec_pre * next_vec).sum().clamp(-1.0, 1.0)
        mismatch = (0.5 * (1.0 - sim_pred)).clamp(0.0, 1.0)

        # ---- Phase C: metabolism (carrier ledgers) ----
        E = self.carriers["energy"]
        E_scale = E.mean().abs().clamp_min(eps)
        cost_rate = (E / E_scale).clamp_min(0.0)
        E = (E * torch.exp(-dt * cost_rate) + dt * intake).clamp_min(0.0)
        self.carriers["energy"] = E

        # ---- Phase D: drift (prototype adaptation) ----
        age = self.carriers["age"]
        inertia = (E * (age + dt)).clamp_min(0.0)
        rate = (intake / (intake + inertia + eps)).clamp(0.0, 1.0)

        ctx_pos = self.carriers["ctx_pos"]
        out_pos = self.carriers["out_pos"]
        ctx_pos = ctx_pos + (dt * rate).unsqueeze(1) * (ctx_vec.unsqueeze(0) - ctx_pos)
        out_pos = out_pos + (dt * rate).unsqueeze(1) * (next_vec.unsqueeze(0) - out_pos)
        self.carriers["ctx_pos"] = _normalize_rows(ctx_pos, eps)
        self.carriers["out_pos"] = _normalize_rows(out_pos, eps)

        # ---- Phase E: heat (friction) ----
        heat = self.carriers["heat"]
        heat = heat + dt * (intake * (d / d_scale.clamp_min(eps)))
        h_scale = heat.mean().clamp_min(eps)
        heat = heat * torch.exp(torch.tensor(-dt, device=heat.device, dtype=heat.dtype) / h_scale)
        self.carriers["heat"] = heat

        # ---- Phase F: age ----
        self.carriers["age"] = self.carriers["age"] + dt

        # ---- Phase G: nucleation (emergence) ----
        # Predict and compare to observation; mismatch accumulates as "unbound mass".
        self.nucleation_mass += float((dt * mismatch).item())
        nucleated = 0
        while self.nucleation_mass >= 1.0:
            self.nucleation_mass -= 1.0
            self._append_carrier(ctx_vec, next_vec, energy=mean_intake.detach())
            nucleated += 1

        self._prune_dead_carriers()

        self.last_debug = {
            "ctx_mass": float(ctx_mass.item()),
            "carriers": self._carrier_count(),
            "mean_intake": float(mean_intake.item()),
            "mismatch": float(mismatch.item()),
            "nucleated": nucleated,
            "w_max": float(w.max().item()),
            "d_scale": float(d_scale.item()) if d_scale.numel() else 0.0,
        }

    # ----------------------------
    # Inference
    # ----------------------------

    def _predict_vector(self) -> torch.Tensor:
        eps = float(self.config.eps)
        if _n_rows(self.particles) == 0 or self._carrier_count() == 0:
            return torch.zeros((self.embed_dim,), dtype=DTYPE_REAL, device=self.device)
        ctx_vec, _ = self._context_vector_and_mass()
        w, _, _ = self._carrier_weights(ctx_vec)
        out_pos = self.carriers["out_pos"]  # [M,D]
        pred_vec = torch.mv(out_pos.t(), w)
        return _normalize_vec(pred_vec, eps)

    def predict_logits(self) -> torch.Tensor:
        """
        Returns logits over vocabulary.

        If an event-horizon index is enabled, logits are sparse: only a candidate
        set receives finite values; others are -inf.
        """
        eps = float(self.config.eps)
        pred_vec = self._predict_vector()

        if pred_vec.abs().sum().item() == 0.0:
            return torch.zeros((self.vocab_size,), dtype=DTYPE_REAL, device=self.device)

        token_pos = self.attractors["position"]

        if self.vocab_index is None:
            sims = torch.mv(token_pos, pred_vec)  # [V]
            s_scale = _safe_std(sims, eps)
            return sims / s_scale

        # Event horizon: only evaluate a derived number of candidates.
        k = int(math.ceil(math.sqrt(float(self.vocab_size))))
        idx, sims = self.vocab_index.topk(pred_vec, k=k)  # [K]
        logits = torch.full((self.vocab_size,), float("-inf"), dtype=DTYPE_REAL, device=self.device)
        if idx.numel() == 0:
            return logits
        s_scale = _safe_std(sims, eps)
        logits[idx] = sims / s_scale
        return logits

    def output_state(self, vocab: Optional[list[str]] = None) -> OutputState:
        eps = float(self.config.eps)
        logits = self.predict_logits()
        probs = torch.softmax(logits, dim=0)

        token_index = int(torch.argmax(probs).item()) if probs.numel() > 0 else None
        token = vocab[token_index] if (vocab is not None and token_index is not None) else None

        # Entropy over full vocabulary (sparse logits contribute 0 mass outside candidates).
        entropy = float(-(probs * torch.log(probs.clamp_min(eps))).sum().item())
        # Confidence as 1 - normalized entropy (dimensionless)
        ent_norm = entropy / max(eps, math.log(float(self.vocab_size + 1)))
        confidence = float(max(0.0, 1.0 - ent_norm))

        meta = {
            "carriers": self._carrier_count(),
            "nucleation_mass": float(self.nucleation_mass),
            "event_horizon": self.vocab_index is not None,
            "index_bits": getattr(self.vocab_index, "total_bits", None) if self.vocab_index is not None else None,
            **self.last_debug,
        }

        return OutputState(
            logits=logits,
            probs=probs,
            token_index=token_index,
            token=token,
            meta={"entropy": entropy, "confidence": confidence, **meta},
        )

    def metrics(self) -> SemanticMetrics:
        M = self._carrier_count()
        if M == 0:
            return SemanticMetrics(
                carriers=0,
                nucleation_mass=float(self.nucleation_mass),
                carrier_energy_mean=0.0,
                carrier_heat_mean=0.0,
                last_debug=dict(self.last_debug),
            )
        e = self.carriers["energy"]
        h = self.carriers["heat"]
        return SemanticMetrics(
            carriers=M,
            nucleation_mass=float(self.nucleation_mass),
            carrier_energy_mean=float(e.mean().item()),
            carrier_heat_mean=float(h.mean().item()),
            last_debug=dict(self.last_debug),
        )