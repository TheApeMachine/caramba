"""
Semantic Manifold: LLM Domain with Thermodynamic Grammar

Implements Bond Topology: Concepts are connected by a transition matrix.
This replaces "Transformer Attention" with "Thermodynamic Flow".

Note: This project does not use backpropagation. Learning is thermodynamic:
bonds survive via energy flow and metabolic maintenance, or starve and vanish.
"""

import math
import torch
from tensordict import TensorDict
from typing import Optional
from physics import ThermodynamicEngine, PhysicsConfig, DTYPE_REAL, OutputState



class SemanticManifold(ThermodynamicEngine):
    """
    The 'Thinking' Engine.
    Implements Bond Topology: Concepts are connected by a transition matrix.
    
    Key Innovation:
    - Grammar is implemented as energy flow through transition_matrix
    - "The" (high energy) -> flows to -> "Dog" (lowers activation energy)
    - This creates sequence-aware predictions, not just "bag of words"
    """
    
    def __init__(
        self,
        config: PhysicsConfig,
        device: torch.device,
        embed_dim: int,
        vocab_size: int,
        max_bonds: Optional[int] = None,
        num_modes: int = 1,
    ):
        super().__init__(config, device)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_bonds = max_bonds
        self.num_modes = max(1, int(num_modes))
        self.halt_mass = 0.0
        self.last_confidence = 0.0
        self.last_entropy = None
        self.last_mode_weights = None
        self.last_mode_entropy = None
        
        # --- THE GRAMMAR PHYSICS ---
        # Transition bonds are not learned by heuristics.
        # They emerge from the geometry of attractors and excitation flow.
        self.transition_matrix = torch.zeros(vocab_size, vocab_size, dtype=DTYPE_REAL, device=device)
        # Each bond maintains its own energy ledger (metabolic budget).
        self.bond_energy = torch.zeros(vocab_size, vocab_size, dtype=DTYPE_REAL, device=device)
        # Context modes: multiple grammar regimes for higher-order context.
        self.mode_bond_energy = torch.zeros(self.num_modes, vocab_size, vocab_size, dtype=DTYPE_REAL, device=device)
        self.mode_trace = torch.zeros_like(self.mode_bond_energy)
        self.mode_prototypes = torch.randn(self.num_modes, vocab_size, dtype=DTYPE_REAL, device=device)
        self.mode_prototypes = self.mode_prototypes / (self.mode_prototypes.norm(dim=1, keepdim=True) + config.eps)
        
        # Initialize attractors (Concepts)
        # These can be seeded with pretrained embeddings when available.
        if embed_dim >= vocab_size:
            # Deterministic, orthogonal basis when possible
            concept_embeddings = torch.eye(vocab_size, embed_dim, dtype=DTYPE_REAL, device=device)
        else:
            concept_embeddings = torch.randn(vocab_size, embed_dim, dtype=DTYPE_REAL, device=device)
            # Normalize embeddings
            concept_embeddings = concept_embeddings / (concept_embeddings.norm(dim=1, keepdim=True) + config.eps)
        
        self.attractors = TensorDict({
            "id": torch.arange(vocab_size, dtype=torch.int64, device=device),
            "position": concept_embeddings,  # Embeddings
            "energy": torch.zeros(vocab_size, dtype=DTYPE_REAL, device=device),
            "excitation": torch.zeros(vocab_size, dtype=DTYPE_REAL, device=device),  # Short-term memory
            "heat": torch.zeros(vocab_size, dtype=DTYPE_REAL, device=device),
        }, batch_size=[vocab_size])

    def ingest_context(self, embeddings: torch.Tensor):
        """
        Turn text tokens into particles.
        
        Args:
            embeddings: [N, embed_dim] token embeddings
        """
        n = embeddings.shape[0]
        # Energy emerges from sequence position: later tokens get higher energy
        idx = torch.arange(1, n + 1, dtype=DTYPE_REAL, device=self.device)
        energy = idx / (idx.max() + self.config.eps)
        
        self.particles = TensorDict({
            "position": embeddings.to(DTYPE_REAL),
            "energy": energy,
            "heat": torch.zeros(n, dtype=DTYPE_REAL, device=self.device),
        }, batch_size=[n])
        self.halt_mass = 0.0
        self.last_confidence = 0.0
        self.last_entropy = None
        self.last_mode_weights = None
        self.last_mode_entropy = None

    def _decay_context(self) -> None:
        """
        Metabolic decay for context particles (no constants).
        Particles lose energy proportional to the system's current energy scale.
        """
        if self.particles.shape[0] == 0:
            return
        e = self.particles.get("energy")
        scale = torch.mean(e.abs()) + self.config.eps
        decay = torch.exp(-self.config.dt / scale)
        self.particles.set("energy", e * decay)

    def _active_indices(self) -> torch.Tensor:
        """
        Active set derived from system scale (no thresholds):
        keep concepts whose activation exceeds the system mean.
        """
        bond_incoming = self.transition_matrix.sum(dim=0)
        total = self.attractors.get("energy") + self.attractors.get("excitation") + bond_incoming
        mean_total = torch.mean(total)
        idx = torch.nonzero(total >= mean_total, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            _, idx = torch.topk(total, k=1, largest=True)
        return idx

    def _slice_attractors(self, indices: torch.Tensor) -> TensorDict:
        sliced = {}
        for key in self.attractors.keys():
            sliced[key] = self.attractors.get(key)[indices]
        return TensorDict(sliced, batch_size=[int(indices.numel())])

    def _merge_attractors(self, indices: torch.Tensor, updated_subset: TensorDict) -> None:
        for key in updated_subset.keys():
            if key in self.attractors.keys():
                self.attractors.get(key)[indices] = updated_subset.get(key)

    def _context_distribution(self, active_idx: torch.Tensor) -> torch.Tensor:
        """
        Return a context distribution over active attractors.
        This aggregates all particles (weighted by energy/recency),
        so grammar is conditioned on the full context field.
        """
        if self.particles.shape[0] == 0 or active_idx.numel() == 0:
            return torch.empty(0, dtype=DTYPE_REAL, device=self.device)
        p = self.particles.get("position")
        a = self.attractors.get("position")[active_idx]
        p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        sim = torch.mm(p_n, a_n.T)
        dists = 1.0 - sim
        dists_scale = torch.std(dists) + self.config.eps
        sharpness = 1.0 / dists_scale
        weights = torch.softmax(-dists * sharpness, dim=1)  # [N, K]
        p_energy = self.particles.get("energy")
        e_scale = p_energy.sum().abs() + self.config.eps
        p_w = p_energy / e_scale
        return torch.matmul(weights.T, p_w)

    def _mode_weights(self, context_resonance: torch.Tensor, error_scale: float) -> torch.Tensor:
        """
        Compute context-conditioned weights over grammar modes.
        """
        if self.num_modes == 1:
            return torch.ones(1, dtype=DTYPE_REAL, device=self.device)
        scores = torch.mv(self.mode_prototypes, context_resonance)
        s_scale = torch.std(scores) + self.config.eps
        sharpness = 1.0 / s_scale
        weights = torch.softmax(scores * sharpness, dim=0)
        if error_scale > 0.0:
            weights = weights.pow(1.0 + error_scale)
            weights = weights / (weights.sum() + self.config.eps)
        return weights

    def compute_distances(self) -> torch.Tensor:
        """
        Cosine Distance: 1 - CosSim
        Returns: [N_particles, M_attractors] distance matrix
        """
        p = self.particles.get("position")  # [N, D]
        a = self.attractors.get("position")  # [M, D]
        
        # Normalize for cosine
        p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        
        # Cosine similarity: [N, M]
        sim = torch.mm(p_n, a_n.T)
        # Distance = 1 - similarity
        return 1.0 - sim

    def compute_targets(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Particles drift toward the concepts they resonate with.
        Target = weighted average of attractor positions.
        
        Args:
            weights: [N_particles, M_attractors] softmax weights
        Returns:
            [N_particles, embed_dim] target positions
        """
        a_pos = self.attractors.get("position")  # [M, embed_dim]
        return torch.mm(weights, a_pos)  # [N, embed_dim]

    def step_grammar(self):
        """
        The "Ghost Field" Logic.
        
        1. Active concepts pump energy into their grammatical successors.
        2. This biases the physics for the NEXT step.
        
        This implements grammar as energy flow through the transition matrix.
        """
        entropy_before = float(self.entropy().item())
        if self.last_entropy is None:
            self.last_entropy = entropy_before
        error_scale = entropy_before / (entropy_before + 1.0)
        debug = {"entropy_before": entropy_before}
        # 0. Decay context energy (recency fades thermodynamically)
        self._decay_context()

        # 1. Get current activation of concepts
        current_exc = self.attractors.get("excitation")  # [Vocab]
        if self.particles.shape[0] > 0:
            # Context resonance competes with stored excitation (metabolic balance)
            p = self.particles.get("position")
            a_full = self.attractors.get("position")
            p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
            a_n = torch.nn.functional.normalize(a_full, dim=1, eps=self.config.eps)
            sim = torch.mm(p_n, a_n.T)
            context_resonance = sim.sum(dim=0)  # [Vocab]
            c_scale = torch.mean(context_resonance.abs()) + self.config.eps
            e_scale = torch.mean(current_exc.abs()) + self.config.eps
            context_resonance = context_resonance / c_scale
            current_exc = current_exc / e_scale
            # Mutual metabolic pressure
            context_resonance = context_resonance * torch.exp(-self.config.dt / e_scale)
            current_exc = current_exc * torch.exp(-self.config.dt / c_scale)
            current_exc = current_exc + context_resonance

        self.attractors.set("excitation", current_exc)
        mode_weights = self._mode_weights(context_resonance, error_scale)
        self.last_mode_weights = mode_weights
        if self.num_modes > 1:
            mode_entropy = -(mode_weights * torch.log(mode_weights + self.config.eps)).sum()
            self.last_mode_entropy = float(mode_entropy.item())
        else:
            self.last_mode_entropy = 0.0
        active_idx = self._active_indices()
        exc_active = current_exc[active_idx]
        
        exc_scale = torch.mean(exc_active.abs()) + self.config.eps
        debug["exc_scale"] = float(exc_scale.item())
        # 2. Emergent Topology: derive adjacency from attractor geometry
        # No thresholds or external tuning. Similarity defines potential flow.
        a = self.attractors.get("position")[active_idx]
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        sim = torch.mm(a_n, a_n.T)  # [-1, 1]
        adjacency = torch.clamp(sim, min=0.0)  # only constructive alignment
        adjacency.fill_diagonal_(0.0)
        debug["adjacency_mean"] = float(adjacency.mean().item())
        debug["adjacency_nonzero"] = float((adjacency > 0).float().mean().item())

        # 2.5 Heat flow: heat diffuses along adjacency and boosts local excitation
        heat_active = self.attractors.get("heat")[active_idx]
        h_scale = torch.mean(heat_active.abs()) + self.config.eps
        heat_level = h_scale / (h_scale + 1.0)
        debug["heat_scale"] = float(h_scale.item())
        debug["heat_level"] = float(heat_level.item())
        # Heat generation: excitation consumes energy and produces heat
        exc_heat = exc_active / (exc_scale + self.config.eps)
        heat_active = heat_active + self.config.dt * exc_heat
        debug["heat_gen_mean"] = float((self.config.dt * exc_heat).mean().item())
        heat_flow = torch.matmul(adjacency, heat_active.unsqueeze(1)).squeeze(1)
        debug["heat_flow_mean"] = float(heat_flow.mean().item())
        heat_active = heat_active + self.config.dt * (heat_flow / h_scale)
        heat_active = heat_active * torch.exp(-self.config.dt / h_scale)
        self.attractors.get("heat")[active_idx] = heat_active
        # Heat increases entropy by reducing energy utility
        heat_utility = 1.0 / (1.0 + h_scale)
        debug["heat_utility"] = float(heat_utility.item())

        # 3. Metabolic Bonds: transition_matrix is bond energy between concepts.
        # Income: excitation of the source concept gated by adjacency,
        # plus observed context flow (sequence-driven, no tuning).
        # Expense: mean excitation scale (metabolic maintenance).
        income = adjacency * exc_active.unsqueeze(1) * heat_utility
        debug["exc_active_mean"] = float(exc_active.mean().item())
        debug["exc_active_abs_mean"] = float(exc_active.abs().mean().item())
        debug["income_abs_mean"] = float(income.abs().mean().item())
        debug["income_mean"] = float(income.mean().item())
        if self.particles.shape[0] > 1:
            # Context-driven transitions from particle order (energy encodes position)
            p = self.particles.get("position")
            a = self.attractors.get("position")[active_idx]
            p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
            a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
            sim_pa = torch.mm(p_n, a_n.T)
            dists = 1.0 - sim_pa
            dists_scale = torch.std(dists) + self.config.eps
            sharpness = 1.0 / dists_scale
            weights = torch.softmax(-dists * sharpness, dim=1)
            order = torch.argsort(self.particles.get("energy"))
            w_src = weights[order[:-1]]
            w_tgt = weights[order[1:]]
            if "heat" in self.particles.keys():
                p_heat = self.particles.get("heat")[order]
                h_scale_p = torch.mean(p_heat.abs()) + self.config.eps
                heat_w = p_heat / h_scale_p
                w_src = w_src * (1.0 + heat_w[:-1].unsqueeze(1))
                w_tgt = w_tgt * (1.0 + heat_w[1:].unsqueeze(1))
            context_flow = torch.mm(w_src.T, w_tgt)
            flow_scale = torch.mean(context_flow.abs()) + self.config.eps
            context_flow = context_flow / flow_scale
            inc_scale = torch.mean(income.abs()) + self.config.eps
            alpha = flow_scale / (flow_scale + inc_scale)
            income = income * (1.0 - alpha) + context_flow * alpha
            debug["context_flow_mean"] = float(context_flow.mean().item())
            debug["flow_scale"] = float(flow_scale.item())
        # Eligibility traces (local, time-extended credit)
        trace_decay = torch.exp(-self.config.dt * (1.0 + heat_level) / exc_scale)
        trace_cap = torch.mean(income.abs()) + self.config.eps

        combined_tm = torch.zeros_like(income)
        for k in range(self.num_modes):
            tm_k = self.mode_bond_energy[k][active_idx[:, None], active_idx]
            trace_k = self.mode_trace[k][active_idx[:, None], active_idx]
            trace_k = trace_k * trace_decay + income
            trace_k = trace_k.clamp(min=-trace_cap, max=trace_cap)
            mean_tm = torch.mean(tm_k.abs()) + self.config.eps
            # Metabolic cost: proportional + flat, both derived from system scale
            expense = exc_scale * (tm_k / mean_tm + 1.0) * (1.0 + heat_level)
            tm_k = (tm_k + self.config.dt * (income * mode_weights[k] - expense)).clamp(min=0.0)
            if self.max_bonds is not None and tm_k.shape[1] > self.max_bonds:
                _, topk_idx = torch.topk(tm_k, k=self.max_bonds, dim=1)
                mask = torch.zeros_like(tm_k)
                mask.scatter_(1, topk_idx, 1.0)
                tm_k = tm_k * mask
            self.mode_bond_energy[k][active_idx[:, None], active_idx] = tm_k
            self.mode_trace[k][active_idx[:, None], active_idx] = trace_k
            combined_tm = combined_tm + tm_k * mode_weights[k]
        self.bond_energy[active_idx[:, None], active_idx] = combined_tm
        self.transition_matrix[active_idx[:, None], active_idx] = combined_tm

        # 4. Flow Energy: Excitation * Transition Matrix
        grammatical_flow_active = torch.matmul(exc_active.unsqueeze(0), combined_tm).squeeze(0)
        debug["grammatical_flow_mean"] = float(grammatical_flow_active.mean().item())

        # 5. Apply Flow: successors heat up, scaled by system excitation
        heat_gain = heat_active / (h_scale + self.config.eps)
        current_exc[active_idx] = current_exc[active_idx] + grammatical_flow_active * (self.config.dt / exc_scale) * (1.0 + heat_gain)

        # Decay (Cooling) derived from excitation scale
        decay = torch.exp(-self.config.dt / exc_scale)
        current_exc = current_exc * decay
        
        self.attractors.set("excitation", current_exc)
        debug["excitation_mean"] = float(current_exc[active_idx].mean().item())
        debug["heat_mean"] = float(heat_active.mean().item())

        # Self-consistency reinforcement: strengthen traced bonds if entropy drops
        context_dist = self._context_distribution(active_idx)
        grammar_bias = torch.matmul(context_dist.unsqueeze(0), combined_tm).squeeze(0)
        c_scale = torch.mean(context_dist.abs()) + self.config.eps
        g_scale = torch.mean(grammar_bias.abs()) + self.config.eps
        confidence = float((g_scale / (c_scale + g_scale)).item())
        entropy_after = float(self.entropy().item())
        debug["entropy_after"] = entropy_after
        delta = self.last_entropy - entropy_after
        mode_entropy_norm = 0.0
        if self.num_modes > 1:
            mode_entropy_norm = self.last_mode_entropy / (math.log(float(self.num_modes)) + self.config.eps)
        coherence = confidence * (1.0 - mode_entropy_norm)
        if abs(delta) > 0.0:
            combined_tm = torch.zeros_like(combined_tm)
            for k in range(self.num_modes):
                trace_k = self.mode_trace[k][active_idx[:, None], active_idx]
                t_scale = torch.mean(trace_k.abs()) + self.config.eps
                tm_k = self.mode_bond_energy[k][active_idx[:, None], active_idx]
                tm_k = (tm_k + self.config.dt * (delta * coherence / t_scale) * trace_k).clamp(min=0.0)
                if self.max_bonds is not None and tm_k.shape[1] > self.max_bonds:
                    _, topk_idx = torch.topk(tm_k, k=self.max_bonds, dim=1)
                    mask = torch.zeros_like(tm_k)
                    mask.scatter_(1, topk_idx, 1.0)
                    tm_k = tm_k * mask
                self.mode_bond_energy[k][active_idx[:, None], active_idx] = tm_k
                combined_tm = combined_tm + tm_k * mode_weights[k]
            self.bond_energy[active_idx[:, None], active_idx] = combined_tm
            self.transition_matrix[active_idx[:, None], active_idx] = combined_tm
        self.last_confidence = confidence
        self.last_entropy = entropy_after
        self.last_debug = debug

    def thinking_confidence(self) -> float:
        """
        Continuous confidence that grammar dominates context.
        """
        if self.particles.shape[0] == 0:
            return 1.0
        active_idx = self._active_indices()
        context_resonance = self._context_distribution(active_idx)
        tm_active = self.transition_matrix[active_idx[:, None], active_idx]
        grammar_bias = torch.matmul(context_resonance.unsqueeze(0), tm_active).squeeze(0)
        c_scale = torch.mean(context_resonance.abs()) + self.config.eps
        g_scale = torch.mean(grammar_bias.abs()) + self.config.eps
        confidence = g_scale / (c_scale + g_scale)
        return float(confidence.item())

    def thinking_complete(self) -> bool:
        """
        Soft halting: integrate confidence into a continuous stop mass.
        """
        if self.particles.shape[0] == 0:
            return True
        confidence = self.thinking_confidence()
        self.halt_mass = min(1.0, self.halt_mass + (confidence * self.config.dt) / (confidence + self.config.eps))
        return self.halt_mass >= 1.0

    def entropy(self) -> torch.Tensor:
        """Return current prediction entropy (scalar)."""
        if self.particles.shape[0] == 0:
            return torch.tensor(0.0, dtype=DTYPE_REAL, device=self.device)
        logits = self.predict_next()
        probs = torch.softmax(logits, dim=0)
        return -(probs * torch.log(probs + self.config.eps)).sum()

    def predict_next(self) -> torch.Tensor:
        """
        Prediction is no longer just similarity.
        It is Similarity + Grammatical Excitation.
        
        Returns:
            [vocab_size] logits for next token
        """
        # 1. Resonance from current particles (Context) on active set
        active_idx = self._active_indices()
        # Ensure the most recent concept and its strongest successor are active
        p = self.particles.get("position")
        a_full = self.attractors.get("position")
        recent_particle = torch.argmax(self.particles.get("energy"))
        p_recent = p[recent_particle : recent_particle + 1]
        p_recent_n = torch.nn.functional.normalize(p_recent, dim=1, eps=self.config.eps)
        a_full_n = torch.nn.functional.normalize(a_full, dim=1, eps=self.config.eps)
        sim_full = torch.mm(p_recent_n, a_full_n.T).squeeze(0)
        recent_full_idx = torch.argmax(sim_full)
        next_full_idx = torch.argmax(self.transition_matrix[recent_full_idx])
        active_idx = torch.unique(torch.cat([active_idx, recent_full_idx.view(1), next_full_idx.view(1)]))
        
        # 2. Grammatical Expectation (The Ghost Field)
        # Flow excitation through bond topology to predict successors
        tm_active = self.transition_matrix[active_idx[:, None], active_idx]
        # Use full context distribution as grammar driver (not just last token)
        context_resonance = self._context_distribution(active_idx)
        grammar_bias = torch.matmul(context_resonance.unsqueeze(0), tm_active).squeeze(0)  # [K]
        
        # 3. Equal Thermodynamic Footing: normalize by emergent scales
        c_scale = torch.mean(context_resonance.abs()) + self.config.eps
        g_scale = torch.mean(grammar_bias.abs()) + self.config.eps
        context_resonance = context_resonance / c_scale
        grammar_bias = grammar_bias / g_scale

        # 3.5 Metabolic Competition: dominance emerges from energy scales
        dominance = g_scale / (c_scale + g_scale)

        # 4. Combined Logits (scatter back to full vocab)
        subset_logits = context_resonance * (1.0 - dominance) + grammar_bias * dominance
        logits = torch.zeros(self.vocab_size, dtype=DTYPE_REAL, device=self.device) - 10.0
        logits[active_idx] = subset_logits
        return logits

    def output_state(self, vocab: Optional[list[str]] = None) -> OutputState:
        """Return unified output state for semantic prediction."""
        logits = self.predict_next()
        probs = torch.softmax(logits, dim=0)
        token_index = int(torch.argmax(probs).item())
        token = vocab[token_index] if vocab is not None else None
        return OutputState(
            logits=logits,
            probs=probs,
            token_index=token_index,
            token=token,
            meta={"entropy": float(self.entropy().item())},
        )

    def dominance_metrics(self) -> dict[str, float]:
        """Return context/grammar scale and dominance share."""
        if self.particles.shape[0] == 0:
            return {"context_scale": 0.0, "grammar_scale": 0.0, "dominance": 0.0}
        active_idx = self._active_indices()
        p = self.particles.get("position")
        a_full = self.attractors.get("position")
        recent_particle = torch.argmax(self.particles.get("energy"))
        p_recent = p[recent_particle : recent_particle + 1]
        p_recent_n = torch.nn.functional.normalize(p_recent, dim=1, eps=self.config.eps)
        a_full_n = torch.nn.functional.normalize(a_full, dim=1, eps=self.config.eps)
        sim_full = torch.mm(p_recent_n, a_full_n.T).squeeze(0)
        recent_full_idx = torch.argmax(sim_full)
        next_full_idx = torch.argmax(self.transition_matrix[recent_full_idx])
        active_idx = torch.unique(torch.cat([active_idx, recent_full_idx.view(1), next_full_idx.view(1)]))
        context_resonance = self._context_distribution(active_idx)
        tm_active = self.transition_matrix[active_idx[:, None], active_idx]
        grammar_bias = torch.matmul(context_resonance.unsqueeze(0), tm_active).squeeze(0)  # [K]

        c_scale = torch.mean(context_resonance.abs()) + self.config.eps
        g_scale = torch.mean(grammar_bias.abs()) + self.config.eps
        dominance = g_scale / (c_scale + g_scale)
        return {
            "context_scale": float(c_scale.item()),
            "grammar_scale": float(g_scale.item()),
            "dominance": float(dominance.item()),
        }

    def energy_metrics(self) -> dict[str, float]:
        """Return energy accounting across system reservoirs."""
        particles_e = self.particles.get("energy") if self.particles.shape[0] > 0 else torch.empty(0, device=self.device)
        attractor_e = self.attractors.get("energy")
        excitation_e = self.attractors.get("excitation")
        bond_e = self.bond_energy
        mode_e = self.mode_bond_energy if hasattr(self, "mode_bond_energy") else None
        heat_a = self.attractors.get("heat") if "heat" in self.attractors.keys() else None
        heat_p = self.particles.get("heat") if self.particles.shape[0] > 0 and "heat" in self.particles.keys() else None
        p_sum = float(particles_e.sum().item()) if particles_e.numel() > 0 else 0.0
        a_sum = float(attractor_e.sum().item()) if attractor_e.numel() > 0 else 0.0
        e_sum = float(excitation_e.sum().item()) if excitation_e.numel() > 0 else 0.0
        b_sum = float(bond_e.sum().item()) if bond_e.numel() > 0 else 0.0
        m_sum = float(mode_e.sum().item()) if mode_e is not None and mode_e.numel() > 0 else 0.0
        ha_sum = float(heat_a.sum().item()) if heat_a is not None and heat_a.numel() > 0 else 0.0
        hp_sum = float(heat_p.sum().item()) if heat_p is not None and heat_p.numel() > 0 else 0.0
        return {
            "particles_energy": p_sum,
            "attractors_energy": a_sum,
            "excitation_energy": e_sum,
            "bond_energy": b_sum,
            "mode_bond_energy": m_sum,
            "attractor_heat": ha_sum,
            "particle_heat": hp_sum,
            "total_energy": p_sum + a_sum + e_sum + b_sum + m_sum,
        }

    def step_physics(self):
        """
        Active-set physics: only compute interactions for active attractors.
        """
        if self.particles.shape[0] == 0:
            self.t += self.config.dt
            return
        active_idx = self._active_indices()
        if active_idx.numel() == 0:
            self.t += self.config.dt
            return
        full_attractors = self.attractors
        active_attractors = self._slice_attractors(active_idx)
        self.attractors = active_attractors
        super().step_physics()
        updated_subset = self.attractors
        self.attractors = full_attractors
        self._merge_attractors(active_idx, updated_subset)

