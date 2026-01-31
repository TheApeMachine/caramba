"""
Semantic Manifold: LLM Domain with Thermodynamic Grammar

Implements Bond Topology: Concepts are connected by a transition matrix.
This replaces "Transformer Attention" with "Thermodynamic Flow".
"""

import torch
from tensordict import TensorDict
from physics import ThermodynamicEngine, PhysicsConfig

DTYPE_REAL = torch.float32


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
        # A learnable adjacency matrix [Vocab, Vocab]
        # T[i, j] = How strongly Concept i excites Concept j
        # This replaces the "Transformer Attention" with "Thermodynamic Flow"
        self.transition_matrix = torch.zeros(vocab_size, vocab_size, dtype=DTYPE_REAL, device=device)
        
        # Initialize attractors (Concepts)
        # In a real app, these would be pretrained embeddings
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
        # Add simple positional encoding (time decay)
        # Recent tokens have higher initial energy
        energy = torch.linspace(0.5, 1.0, n, dtype=DTYPE_REAL, device=self.device)
        
        self.particles = TensorDict({
            "position": embeddings.to(DTYPE_REAL),
            "energy": energy,
        }, batch_size=[n])

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
        # 1. Get current activation of concepts
        # We use 'excitation' which is a smoothed version of energy
        current_exc = self.attractors.get("excitation")  # [Vocab]
        
        # 2. Flow Energy: Excitation * Transition Matrix
        # If "The" is excited, it sends energy to "Dog", "Cat", etc.
        # [Vocab] @ [Vocab, Vocab] -> [Vocab]
        grammatical_flow = torch.matmul(current_exc.unsqueeze(0), self.transition_matrix).squeeze(0)
        
        # 3. Apply Flow: This heats up the successors
        # They become "magnetic" for the next token prediction
        current_exc = current_exc + grammatical_flow * self.config.transition_flux * self.config.dt
        
        # Decay (Cooling) - prevents infinite accumulation
        current_exc = current_exc * 0.95
        
        self.attractors.set("excitation", current_exc)

    def predict_next(self) -> torch.Tensor:
        """
        Prediction is no longer just similarity.
        It is Similarity + Grammatical Excitation.
        
        Returns:
            [vocab_size] logits for next token
        """
        # 1. Resonance from current particles (Context)
        dists = self.compute_distances()  # [N, Vocab]
        # Sum of particle resonance for each concept
        # Lower distance = higher resonance
        context_resonance = (1.0 - dists).sum(dim=0)  # [Vocab]
        
        # 2. Grammatical Expectation (The Ghost Field)
        grammar_bias = self.attractors.get("excitation")  # [Vocab]
        
        # 3. Combined Logits
        logits = context_resonance + grammar_bias
        return logits

    def learn_transition(self, token_seq: torch.Tensor):
        """
        Hebbian Learning for Grammar.
        If Token A is followed by Token B, strengthen the bond A->B.
        
        Args:
            token_seq: [N] sequence of token IDs
        """
        for i in range(len(token_seq) - 1):
            src = int(token_seq[i].item())
            dst = int(token_seq[i + 1].item())
            # Strengthen connection
            self.transition_matrix[src, dst] += 0.1
        
        # Normalize rows (Probability)
        row_sums = self.transition_matrix.sum(dim=1, keepdim=True) + 1e-6
        self.transition_matrix = self.transition_matrix / row_sums
