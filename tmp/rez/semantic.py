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
    
    def __init__(self, config: PhysicsConfig, device: torch.device, embed_dim: int, vocab_size: int):
        super().__init__(config, device)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # --- THE GRAMMAR PHYSICS ---
        # Transition bonds are not learned by heuristics.
        # They emerge from the geometry of attractors and excitation flow.
        self.transition_matrix = torch.zeros(vocab_size, vocab_size, dtype=DTYPE_REAL, device=device)
        
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
        }, batch_size=[n])

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
        active_idx = self._active_indices()
        exc_active = current_exc[active_idx]
        
        # 2. Emergent Topology: derive adjacency from attractor geometry
        # No thresholds or external tuning. Similarity defines potential flow.
        a = self.attractors.get("position")[active_idx]
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        sim = torch.mm(a_n, a_n.T)  # [-1, 1]
        adjacency = torch.clamp(sim, min=0.0)  # only constructive alignment
        adjacency.fill_diagonal_(0.0)

        # 3. Metabolic Bonds: transition_matrix is bond energy between concepts.
        # Income: excitation of the source concept gated by adjacency,
        # plus observed context flow (sequence-driven, no tuning).
        # Expense: mean excitation scale (metabolic maintenance).
        exc_scale = torch.mean(exc_active.abs()) + self.config.eps
        tm = self.transition_matrix[active_idx[:, None], active_idx]
        income = adjacency * exc_active.unsqueeze(1)
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
            context_flow = torch.mm(w_src.T, w_tgt)
            flow_scale = torch.mean(context_flow.abs()) + self.config.eps
            context_flow = context_flow / flow_scale
            income = context_flow
        expense = tm * exc_scale
        tm = (tm + self.config.dt * (income - expense)).clamp(min=0.0)
        self.transition_matrix[active_idx[:, None], active_idx] = tm

        # 4. Flow Energy: Excitation * Transition Matrix
        grammatical_flow_active = torch.matmul(exc_active.unsqueeze(0), tm).squeeze(0)

        # 5. Apply Flow: successors heat up, scaled by system excitation
        current_exc[active_idx] = current_exc[active_idx] + grammatical_flow_active * (self.config.dt / exc_scale)

        # Decay (Cooling) derived from excitation scale
        decay = torch.exp(-self.config.dt / exc_scale)
        current_exc = current_exc * decay
        
        self.attractors.set("excitation", current_exc)

    def thinking_complete(self) -> bool:
        """
        Decide if the system has "settled" based on metabolic balance.
        Thinking completes when grammar energy meets or exceeds context energy.
        """
        if self.particles.shape[0] == 0:
            return True
        # Mirror predict_next to compare context vs grammar scales
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
        active_attractors = self._slice_attractors(active_idx)

        a = active_attractors.get("position")
        p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        sim = torch.mm(p_n, a_n.T)
        dists = 1.0 - sim
        dists_scale = torch.std(dists) + self.config.eps
        sharpness = 1.0 / dists_scale
        weights = torch.softmax(-dists * sharpness, dim=1)
        driver_weights = weights[recent_particle]
        context_resonance = driver_weights
        driver_idx = torch.argmax(driver_weights)
        driver = torch.zeros_like(driver_weights)
        driver[driver_idx] = 1.0
        tm_active = self.transition_matrix[active_idx[:, None], active_idx]
        grammar_bias = torch.matmul(driver.unsqueeze(0), tm_active).squeeze(0)

        c_scale = torch.mean(context_resonance.abs()) + self.config.eps
        g_scale = torch.mean(grammar_bias.abs()) + self.config.eps
        return bool(g_scale >= c_scale)

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
        p = self.particles.get("position")
        a_full = self.attractors.get("position")
        # Ensure the most recent concept and its strongest successor are active
        recent_particle = torch.argmax(self.particles.get("energy"))
        p_recent = p[recent_particle : recent_particle + 1]
        p_recent_n = torch.nn.functional.normalize(p_recent, dim=1, eps=self.config.eps)
        a_full_n = torch.nn.functional.normalize(a_full, dim=1, eps=self.config.eps)
        sim_full = torch.mm(p_recent_n, a_full_n.T).squeeze(0)
        recent_full_idx = torch.argmax(sim_full)
        next_full_idx = torch.argmax(self.transition_matrix[recent_full_idx])
        active_idx = torch.unique(torch.cat([active_idx, recent_full_idx.view(1), next_full_idx.view(1)]))
        active_attractors = self._slice_attractors(active_idx)

        a = active_attractors.get("position")
        p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        sim = torch.mm(p_n, a_n.T)
        dists = 1.0 - sim
        
        # 2. Grammatical Expectation (The Ghost Field)
        # Flow excitation through bond topology to predict successors
        tm_active = self.transition_matrix[active_idx[:, None], active_idx]
        # Use most recent particle as the grammar driver (energy encodes recency)
        dists_scale = torch.std(dists) + self.config.eps
        sharpness = 1.0 / dists_scale
        weights = torch.softmax(-dists * sharpness, dim=1)  # [N, K]
        recent_idx = int(torch.argmax(self.particles.get("energy")).item())
        driver_weights = weights[recent_idx]  # [K]
        # Context resonance is the current concept (most recent particle)
        context_resonance = driver_weights
        driver_idx = int(torch.argmax(driver_weights).item())
        driver = torch.zeros_like(driver_weights)
        driver[driver_idx] = 1.0
        grammar_bias = torch.matmul(driver.unsqueeze(0), tm_active).squeeze(0)  # [K]
        
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
        active_attractors = self._slice_attractors(active_idx)

        a = active_attractors.get("position")
        p_n = torch.nn.functional.normalize(p, dim=1, eps=self.config.eps)
        a_n = torch.nn.functional.normalize(a, dim=1, eps=self.config.eps)
        sim = torch.mm(p_n, a_n.T)
        dists = 1.0 - sim
        dists_scale = torch.std(dists) + self.config.eps
        sharpness = 1.0 / dists_scale
        weights = torch.softmax(-dists * sharpness, dim=1)  # [N, K]
        recent_idx = int(torch.argmax(self.particles.get("energy")).item())
        driver_weights = weights[recent_idx]  # [K]
        context_resonance = driver_weights
        driver_idx = int(torch.argmax(driver_weights).item())
        driver = torch.zeros_like(driver_weights)
        driver[driver_idx] = 1.0
        tm_active = self.transition_matrix[active_idx[:, None], active_idx]
        grammar_bias = torch.matmul(driver.unsqueeze(0), tm_active).squeeze(0)  # [K]

        c_scale = torch.mean(context_resonance.abs()) + self.config.eps
        g_scale = torch.mean(grammar_bias.abs()) + self.config.eps
        dominance = g_scale / (c_scale + g_scale)
        return {
            "context_scale": float(c_scale.item()),
            "grammar_scale": float(g_scale.item()),
            "dominance": float(dominance.item()),
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

