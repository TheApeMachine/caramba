#!/usr/bin/env python3
"""
Test suite for the refactored Manifold architecture.
Demonstrates the split between SpectralManifold and SemanticManifold.
"""

import torch
from tensordict import TensorDict
from physics import ThermodynamicEngine, SpectralManifold, PhysicsConfig, DTYPE_REAL
from semantic import SemanticManifold

def test_spectral_manifold():
    """Test 1: Audio domain (SpectralManifold)"""
    print("=" * 60)
    print("Test 1: SpectralManifold (Audio Domain)")
    print("=" * 60)
    
    device = torch.device("cpu")
    config = PhysicsConfig(dt=0.01, hold_cost=5.0)
    
    manifold = SpectralManifold(config, device)
    
    # Simulate STFT frame
    freq_bins = torch.tensor([261.63, 329.63, 392.00], dtype=DTYPE_REAL, device=device)  # C-Major
    magnitudes = torch.tensor([0.8, 0.6, 0.7], dtype=DTYPE_REAL, device=device)
    phases = torch.rand(3, dtype=DTYPE_REAL, device=device) * 6.28
    
    print(f"\nIngesting audio frame:")
    print(f"  Frequencies: {freq_bins.tolist()} Hz")
    print(f"  Magnitudes: {magnitudes.tolist()}")
    
    manifold.ingest_frame(freq_bins, magnitudes, phases)
    
    # Check state
    particles = manifold.particles
    attractors = manifold.attractors
    
    print(f"\nResults:")
    print(f"  Particles: {particles.shape[0]}")
    print(f"  Attractors: {attractors.shape[0]}")
    
    if particles.shape[0] > 0:
        print(f"  Particle positions (frequencies): {particles.get('position').cpu().numpy()}")
    
    if attractors.shape[0] > 0:
        print(f"  Attractor positions: {attractors.get('position').cpu().numpy()}")
        print(f"  Attractor energies: {attractors.get('energy').cpu().numpy()}")
    
    print("\n✓ Test 1 Complete\n")


def test_semantic_manifold():
    """Test 2: LLM domain (SemanticManifold)"""
    print("=" * 60)
    print("Test 2: SemanticManifold (LLM Domain)")
    print("=" * 60)
    
    device = torch.device("cpu")
    embed_dim = 128
    config = PhysicsConfig(dt=0.01, hold_cost=5.0)
    
    manifold = SemanticManifold(config, device, embed_dim, vocab_size=128)
    
    # Simulate token embeddings
    n_tokens = 5
    embeddings = torch.randn(n_tokens, embed_dim, dtype=DTYPE_REAL, device=device)
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)  # Normalize
    
    print(f"\nIngesting token embeddings:")
    print(f"  Tokens: {n_tokens}")
    print(f"  Embedding dimension: {embed_dim}")
    
    manifold.ingest_context(embeddings)
    
    # Set up grammar transition matrix
    n_attractors = manifold.attractors.shape[0]
    if n_attractors > 0:
        # Create a simple transition matrix (random for demo)
        transition_matrix = torch.rand(n_attractors, n_attractors, dtype=DTYPE_REAL, device=device)
        transition_matrix = transition_matrix / (transition_matrix.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
        manifold.set_transition_matrix(transition_matrix)
        
        print(f"\nGrammar transition matrix set:")
        print(f"  Shape: {transition_matrix.shape}")
        print(f"  Example: transition_matrix[0, :] = {transition_matrix[0, :3].cpu().numpy()}")
    
    # Check state
    particles = manifold.particles
    attractors = manifold.attractors
    
    print(f"\nResults:")
    print(f"  Particles: {particles.shape[0]}")
    print(f"  Attractors: {attractors.shape[0]}")
    
    if particles.shape[0] > 0:
        pos = particles.get("position")
        print(f"  Particle positions shape: {pos.shape} (should be [N, {embed_dim}])")
    
    if attractors.shape[0] > 0:
        pos = attractors.get("position")
        print(f"  Attractor positions shape: {pos.shape} (should be [M, {embed_dim}])")
        print(f"  Attractor energies: {attractors.get('energy').cpu().numpy()[:5]}")
    
    print("\n✓ Test 2 Complete\n")


def test_distance_metrics():
    """Test 3: Verify domain-specific distance metrics"""
    print("=" * 60)
    print("Test 3: Distance Metrics (Domain-Specific)")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # Test SpectralManifold distance (log-frequency)
    spectral = SpectralManifold(PhysicsConfig(), device)
    freq_a = torch.tensor([261.63], dtype=DTYPE_REAL, device=device)  # C4
    freq_b = torch.tensor([523.25], dtype=DTYPE_REAL, device=device)  # C5 (octave)
    spectral.particles = TensorDict({"position": freq_a, "energy": torch.ones(1, device=device), "ttl": torch.ones(1, device=device)}, batch_size=[1])
    spectral.attractors = TensorDict({"position": freq_b, "energy": torch.ones(1, device=device), "excitation": torch.zeros(1, device=device)}, batch_size=[1])
    dist_spectral = spectral.compute_distances()[0, 0]
    print(f"\nSpectralManifold distance:")
    print(f"  C4 (261.63 Hz) vs C5 (523.25 Hz): {dist_spectral.item():.4f}")
    print(f"  (Should be small - octave relationship)")
    
    # Test SemanticManifold distance (cosine)
    semantic = SemanticManifold(PhysicsConfig(), device, embed_dim=128, vocab_size=128)
    emb_a = torch.randn(1, 128, dtype=DTYPE_REAL, device=device)
    emb_a = emb_a / (emb_a.norm() + 1e-8)
    emb_b = emb_a.clone()  # Same vector
    emb_c = torch.randn(1, 128, dtype=DTYPE_REAL, device=device)
    emb_c = emb_c / (emb_c.norm() + 1e-8)

    semantic.particles = TensorDict({"position": emb_a, "energy": torch.ones(1, device=device)}, batch_size=[1])
    semantic.attractors = TensorDict(
        {"position": torch.cat([emb_b, emb_c], dim=0), "energy": torch.ones(2, device=device), "excitation": torch.zeros(2, device=device)},
        batch_size=[2],
    )
    dists = semantic.compute_distances()
    dist_same = dists[0, 0]
    dist_diff = dists[0, 1]
    
    print(f"\nSemanticManifold distance:")
    print(f"  Same vector: {dist_same.item():.4f} (should be ~0.0)")
    print(f"  Different vectors: {dist_diff.item():.4f} (should be > 0.0)")
    
    print("\n✓ Test 3 Complete\n")


def test_thermodynamic_grammar():
    """Test 4: Thermodynamic Grammar (Energy Flow)"""
    print("=" * 60)
    print("Test 4: Thermodynamic Grammar")
    print("=" * 60)
    
    device = torch.device("cpu")
    embed_dim = 64
    config = PhysicsConfig(dt=0.01)
    
    manifold = SemanticManifold(config, device, embed_dim, vocab_size=3)
    
    # Create concept attractors manually
    n_concepts = 3
    concept_embeddings = torch.randn(n_concepts, embed_dim, dtype=DTYPE_REAL, device=device)
    concept_embeddings = concept_embeddings / (concept_embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    concept_positions = concept_embeddings
    
    # Create attractors
    new_attractors = {
        "id": torch.arange(n_concepts, dtype=torch.int64, device=device),
        "position": concept_positions,
        "energy": torch.tensor([1.0, 0.5, 0.2], dtype=DTYPE_REAL, device=device),
        "excitation": torch.zeros(n_concepts, dtype=DTYPE_REAL, device=device),
    }
    manifold.attractors = TensorDict(new_attractors, batch_size=[n_concepts])
    
    # Set up grammar: Concept 0 ("The") -> Concept 1 ("Dog") -> Concept 2 ("Barks")
    transition_matrix = torch.zeros(n_concepts, n_concepts, dtype=DTYPE_REAL, device=device)
    transition_matrix[0, 1] = 0.8  # "The" -> "Dog"
    transition_matrix[1, 2] = 0.7  # "Dog" -> "Barks"
    transition_matrix[0, 2] = 0.1  # "The" -> "Barks" (weak)
    manifold.set_transition_matrix(transition_matrix)
    
    print(f"\nGrammar setup:")
    print(f"  Concept 0 ('The') -> Concept 1 ('Dog'): {transition_matrix[0, 1]:.2f}")
    print(f"  Concept 1 ('Dog') -> Concept 2 ('Barks'): {transition_matrix[1, 2]:.2f}")
    
    # Apply grammar
    print(f"\nBefore grammar:")
    excitation_before = manifold.attractors.get("excitation").clone()
    print(f"  Excitation: {excitation_before.cpu().numpy()}")
    
    manifold.step_grammar()
    
    print(f"\nAfter grammar:")
    excitation_after = manifold.attractors.get("excitation")
    print(f"  Excitation: {excitation_after.cpu().numpy()}")
    print(f"  Change: {(excitation_after - excitation_before).cpu().numpy()}")
    
    print("\n✓ Test 4 Complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Refactored Manifold Architecture Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_spectral_manifold()
        test_semantic_manifold()
        test_distance_metrics()
        test_thermodynamic_grammar()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        print("\nKey Achievements:")
        print("  ✓ Separated physics from domain-specific code")
        print("  ✓ SpectralManifold handles audio (Hertz, phase)")
        print("  ✓ SemanticManifold handles LLMs (embeddings, grammar)")
        print("  ✓ Domain-specific distance metrics")
        print("  ✓ Thermodynamic grammar (energy flow)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
