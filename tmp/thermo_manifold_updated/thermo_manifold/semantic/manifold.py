from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..core.config import PhysicsConfig
from ..core.state import BatchState
from ..physics.engine import ThermodynamicEngine
from .bond_graph import SparseBondGraph


@dataclass
class SemanticOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    token_index: int
    token: Optional[str]
    meta: Dict[str, Any]


class SemanticManifold(ThermodynamicEngine):
    """Thermodynamic grammar on a sparse bond graph (no dense V×V matrices)."""

    def __init__(self, config: PhysicsConfig, device: torch.device, *, vocab: List[str], embed_dim: Optional[int] = None):
        super().__init__(config, device)
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        self.embed_dim = int(embed_dim if embed_dim is not None else self.vocab_size)

        pos = self._init_embeddings(self.vocab_size, self.embed_dim, eps=self.cfg.eps, device=device)
        self.attractors = BatchState(
            {
                "id": torch.arange(self.vocab_size, device=device, dtype=torch.long),
                "position": pos,
                "energy": torch.zeros(self.vocab_size, device=device, dtype=torch.float32),
                "excitation": torch.zeros(self.vocab_size, device=device, dtype=torch.float32),
                "heat": torch.zeros(self.vocab_size, device=device, dtype=torch.float32),
            }
        )

        self.graph = SparseBondGraph(self.vocab_size, device=device, dtype=torch.float32, eps=self.cfg.eps)

        # Thinking / halting state
        self.halt_mass = 0.0
        self.last_entropy: Optional[float] = None
        self.last_confidence: float = 0.0
        self.last_debug: Dict[str, float] = {}

    def total_energy(self) -> torch.Tensor:
        """Homeostasis energy for semantic dynamics.

        Excitation is treated as the dominant energetic quantity for the grammar.
        """
        total = super().total_energy()
        exc = self.attractors.get("excitation")
        return total + exc.abs().sum().to(torch.float32)

    @staticmethod
    def _init_embeddings(vocab_size: int, embed_dim: int, *, eps: float, device: torch.device) -> torch.Tensor:
        if embed_dim >= vocab_size:
            emb = torch.eye(vocab_size, embed_dim, device=device, dtype=torch.float32)
        else:
            emb = torch.randn(vocab_size, embed_dim, device=device, dtype=torch.float32)
            emb = emb / (emb.norm(dim=1, keepdim=True) + eps)
        return emb

    # ----------------------------
    # Ingest
    # ----------------------------

    def ingest_ids(self, ids: torch.Tensor) -> None:
        """Ingest a token-id context as particles."""
        ids = ids.to(device=self.device, dtype=torch.long).flatten()
        n = int(ids.numel())
        if n == 0:
            self.particles = BatchState.empty()
            return

        idx = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        energy = idx / (idx.max() + self.cfg.eps)

        pos = self.attractors.get("position")[ids]
        self.particles = BatchState(
            {
                "id": ids,
                "position": pos,
                "energy": energy,
                "heat": torch.zeros(n, device=self.device, dtype=torch.float32),
            }
        )

        # Energy enters the semantic manifold via observation (context excitation).
        exc = self.attractors.get("excitation")
        e_norm = energy / (energy.sum() + self.cfg.eps)
        exc.index_add_(0, ids, e_norm)
        self.attractors.set("excitation", exc)

        # Reset thinking state for the new observation.
        self.halt_mass = 0.0
        self.last_entropy = None
        self.last_confidence = 0.0
        self.last_debug = {}

    # ----------------------------
    # Grammar phases
    # ----------------------------

    def update_topology(self) -> None:
        """Topology update phase: nucleate/strengthen edges from observed order."""
        if self.particles.n < 2:
            return
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        ids = self.particles.get("id")
        e = self.particles.get("energy")
        e_scale = e.abs().mean() + eps

        # Nucleation mass emerges from current context energy scale.
        mass = torch.tensor(dt, device=self.device, dtype=torch.float32) * (e.mean() / e_scale)
        self.graph.add_path(ids, mass)

    def run_metabolism(self, active_src: torch.Tensor, *, ratio: torch.Tensor) -> None:
        """Metabolism phase: update bond masses and traces for active sources."""
        if active_src.numel() == 0 or self.graph.num_edges == 0:
            return
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        exc = self.attractors.get("excitation")
        heat = self.attractors.get("heat")
        exc_scale = exc[active_src].abs().mean() + eps

        batch = self.graph.batch_edges(active_src)
        if batch is None:
            return

        # Outgoing normalization per source (local).
        out_sum = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
        out_sum.index_add_(0, batch.src, batch.w)
        w_norm = batch.w / (out_sum[batch.src] + eps)

        # Flow-based usage.
        use = exc[batch.src] * w_norm

        # Heat reduces usable energy (entropy).
        h_level = heat[batch.src].abs().mean()
        heat_utility = 1.0 / (1.0 + h_level + eps)
        income = use * heat_utility

        # Trace: local time-extended credit.
        trace = batch.trace
        trace_scale = trace.abs().mean() + eps
        trace_decay = torch.exp(-dt * ratio / (exc_scale + trace_scale))
        trace_new = trace * trace_decay + income

        # Cost: proportional decay (edges starve without use).
        cost = ratio * exc_scale * batch.w / (batch.w.abs().mean() + eps)
        w_new = (batch.w + dt * (income - cost)).clamp(min=0.0)

        # Write back.
        eidx: List[int] = []
        for s in active_src.tolist():
            eidx.extend(self.graph.edges_from(int(s)))
        if not eidx:
            return
        eidx_t = torch.tensor(eidx, device=self.device, dtype=torch.long)
        self.graph.update_edges(eidx_t, w_new, trace_new)

        # Prune weak edges without fixed thresholds.
        self.graph.prune_by_src_mean(active_src)

        self.last_debug.update(
            {
                "exc_scale": float(exc_scale.item()),
                "heat_level": float(h_level.item()),
                "income_mean": float(income.mean().item()),
                "cost_mean": float(cost.mean().item()),
                "edges_active": float(len(eidx)),
            }
        )

    def propagate_flow(self, active_src: torch.Tensor, *, ratio: torch.Tensor) -> None:
        """Flow propagation phase: push excitation through bonds."""
        if active_src.numel() == 0 or self.graph.num_edges == 0:
            return
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        exc = self.attractors.get("excitation")
        heat = self.attractors.get("heat")
        exc_scale = exc[active_src].abs().mean() + eps

        # Sparse source distribution from excitation on active sources.
        dist = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
        dist[active_src] = exc[active_src].clamp(min=0.0)
        dist = dist / (dist.sum() + eps)

        flow = self.graph.flow_from_distribution(dist)

        # Heat generation: using excitation produces heat.
        nnz = torch.count_nonzero(flow)
        flow_scale = flow.abs().sum() / (nnz.to(flow.dtype) + eps)
        heat = heat + dt * (flow.abs() / (flow_scale + eps))

        # Homeostatic excitation damping.
        exc = (exc + dt * flow) * torch.exp(-dt * ratio / (exc_scale + eps))

        self.attractors.set("excitation", exc)
        self.attractors.set("heat", heat)

        self.last_debug.update(
            {
                "flow_mean": float(flow.mean().item()),
                "exc_mean": float(exc.mean().item()),
                "heat_mean": float(heat.mean().item()),
            }
        )

    def step_grammar(self) -> None:
        """One grammar step, decomposed into phases for debuggability."""
        if self.particles.n == 0:
            return

        ratio = self._homeostasis_ratio().to(torch.float32)

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        # Continuous observation: while particles exist, the manifold receives energy.
        exc = self.attractors.get("excitation")
        ids = self.particles.get("id")
        e = self.particles.get("energy").clamp(min=0.0)
        exc.index_add_(0, ids, torch.tensor(dt, device=self.device) * (e / (e.sum() + eps)))
        self.attractors.set("excitation", exc)

        # Phase 1: topology
        self.update_topology()

        # Active sources: context tokens (locality without global thresholds).
        active_src = torch.unique(self.particles.get("id"))

        # Phase 2: metabolism
        self.run_metabolism(active_src, ratio=ratio)

        # Phase 3: propagation
        self.propagate_flow(active_src, ratio=ratio)

        ent = float(self.entropy().item())
        if self.last_entropy is None:
            self.last_entropy = ent
        self.last_debug["entropy"] = ent
        self.last_confidence = self.thinking_confidence()

    # ----------------------------
    # Readout
    # ----------------------------

    def _context_distribution(self) -> torch.Tensor:
        eps = self.cfg.eps
        dist = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
        ids = self.particles.get("id")
        e = self.particles.get("energy").clamp(min=0.0)
        dist.index_add_(0, ids, e / (e.sum() + eps))
        return dist

    def predict_next(self) -> torch.Tensor:
        eps = self.cfg.eps
        ctx = self._context_distribution()
        gram = self.graph.flow_from_distribution(ctx)

        c_scale = ctx.abs().sum() + eps
        g_scale = gram.abs().sum()

        # Readout prioritizes *forward* flow when grammar has support; otherwise fall back to context.
        if float(g_scale.item()) > float(eps):
            return gram / (g_scale + eps)
        return ctx / c_scale

    def entropy(self) -> torch.Tensor:
        logits = self.predict_next()
        probs = torch.softmax(logits, dim=0)
        return -(probs * torch.log(probs + self.cfg.eps)).sum()

    def thinking_confidence(self) -> float:
        ctx = self._context_distribution()
        gram = self.graph.flow_from_distribution(ctx)
        c = ctx.abs().sum() + self.cfg.eps
        g = gram.abs().sum()
        # Confidence is the amount of forward signal available relative to the observation mass.
        conf = (g / (c + self.cfg.eps)).clamp(min=0.0, max=1.0)
        return float(conf.item())

    def thinking_complete(self) -> bool:
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        conf = self.thinking_confidence()
        self.halt_mass = min(1.0, self.halt_mass + (conf * dt) / (conf + eps))
        return self.halt_mass >= 1.0

    def output_state(self) -> SemanticOutput:
        logits = self.predict_next()
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.argmax(probs).item())
        tok = self.vocab[idx] if 0 <= idx < len(self.vocab) else None
        meta = {
            "entropy": float(self.entropy().item()),
            "confidence": float(self.last_confidence),
            **self.last_debug,
        }
        return SemanticOutput(logits=logits, probs=probs, token_index=idx, token=tok, meta=meta)
