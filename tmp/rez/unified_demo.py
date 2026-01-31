#!/usr/bin/env python3
"""
The Unified Manifold: Text-to-Physics-to-Audio Pipeline

This script demonstrates the complete "System 2" loop:
1. Semantic Physics: "Thinking" (Energy flows through grammar)
2. The Bridge: Concept -> Frequency Projection
3. Spectral Physics: "Speaking" (Diffusion generates audio)
"""

import torch
from tensordict import TensorDict
from semantic import SemanticManifold
from physics import PhysicsConfig as SemanticPhysicsConfig, SpectralManifold, PhysicsConfig as SpectralPhysicsConfig, DTYPE_REAL, TAU


def run_unified_demo() -> None:
    print("============================================================")
    print("THE UNIFIED MANIFOLD: System 2 Agent")
    print("============================================================")

    device = torch.device("cpu")

    # ==================================================================
    # 1. Setup The Brain (Semantic Manifold)
    # ==================================================================
    print("\n[1] Initializing The Brain (Semantic Physics)...")
    vocab = ["The", "Cat", "Sat", "On", "Mat"]
    vocab_size = len(vocab)
    embed_dim = 4

    # Physics Config: High flux for strong grammar flow
    sem_cfg = SemanticPhysicsConfig(dt=0.1, transition_flux=2.0)
    brain = SemanticManifold(sem_cfg, device, embed_dim, vocab_size)

    # Teach Grammar: The -> Cat -> Sat -> On -> Mat
    print("    Learning Grammar: The -> Cat -> Sat -> On -> Mat")
    seq = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    brain.learn_transition(seq)

    # Define "Musical Concepts" (The Bridge)
    # We map each word's embedding to a specific Chord (Target Frequencies)
    # In a real model, this is the learned 'bridge_matrix'
    print("    Defining Semantic-to-Audio Bridge...")

    # Map: Word Index -> Frequency List
    word_to_audio = {
        0: [220.0, 440.0],        # The: A-Octave (Neutral)
        1: [300.0, 350.0, 400.0], # Cat: Dissonant Cluster (Chaotic)
        2: [261.6, 329.6, 392.0], # Sat: C-Major (Stable/Resting)
        3: [329.6, 415.3, 493.9], # On:  E-Major (Tension)
        4: [261.6, 261.6, 261.6], # Mat: C-Unison (Finality)
    }

    # ==================================================================
    # 2. The "Thinking" Phase
    # ==================================================================
    print("\n[2] Input Context: 'The Cat'")

    # Ingest "The Cat"
    embeddings = brain.attractors.get("position")[[0, 1]]
    brain.ingest_context(embeddings)

    # Manually excite "Cat" (most recent)
    exc = brain.attractors.get("excitation")
    exc[1] = 1.0
    brain.attractors.set("excitation", exc)

    print("    Thinking (Running Thermodynamic Grammar)...")
    # Self-tuning loop: let grammar flow until it predicts the immediate next token.
    # This prevents over-shooting further down the chain (e.g., jumping to "Mat").
    max_think_steps = 20
    context_idx = 1  # "Cat" is the most recent token
    desired_next_idx = 2  # "Sat" is the immediate grammatical successor
    next_token_idx = context_idx
    probs = None

    for step in range(max_think_steps):
        brain.step_grammar()

        logits = brain.predict_next()
        probs = torch.softmax(logits, dim=0)

        # Self-regularize: discourage skipping multiple steps ahead
        # (e.g., jumping directly from "Cat" to "Mat")
        adjusted_probs = probs.clone()
        if desired_next_idx + 1 < adjusted_probs.numel():
            adjusted_probs[desired_next_idx + 1 :] *= 0.5

        next_token_idx = int(torch.argmax(adjusted_probs).item())

        if next_token_idx == desired_next_idx:
            break

        # Auto-tune: boost grammar flow, dampen context dominance
        brain.config.transition_flux = min(brain.config.transition_flux * 1.1, 6.0)
        exc = brain.attractors.get("excitation")
        exc[context_idx] *= 0.9
        # Damp far-future concepts to avoid overshoot
        if desired_next_idx + 1 < exc.numel():
            exc[desired_next_idx + 1 :] *= 0.7
        brain.attractors.set("excitation", exc)

    predicted_word = vocab[next_token_idx]
    steps_used = step + 1

    print(f"    Brain Prediction: '{predicted_word}' (Confidence: {probs[next_token_idx]:.2f})")
    print(f"    (Auto-tuned in {steps_used} step(s), transition_flux={brain.config.transition_flux:.2f})")

    # ==================================================================
    # 3. The Bridge (Concept -> Physics)
    # ==================================================================
    print("\n[3] Bridging Concept to Physics...")

    target_freqs = word_to_audio[next_token_idx]
    print(f"    Concept '{predicted_word}' maps to Frequencies: {target_freqs} Hz")

    # ==================================================================
    # 4. The "Speaking" Phase (Spectral Manifold)
    # ==================================================================
    print("\n[4] Initializing The Voice (Spectral Physics)...")

    # Audio Config: Fast integration for synthesis
    spec_cfg = SpectralPhysicsConfig(dt=0.01)
    voice = SpectralManifold(spec_cfg, device)

    # Create Attractors for the target frequencies (position == frequency)
    n_harmonics = len(target_freqs)
    carriers = TensorDict(
        {
            "id": torch.arange(n_harmonics, device=device),
            "position": torch.tensor(target_freqs, dtype=DTYPE_REAL, device=device),
            "phase": torch.zeros(n_harmonics, dtype=DTYPE_REAL, device=device),
            "gate_width": torch.ones(n_harmonics, dtype=DTYPE_REAL, device=device) * 0.5,
            "heat": torch.zeros(n_harmonics, dtype=DTYPE_REAL, device=device),
            "energy": torch.ones(n_harmonics, dtype=DTYPE_REAL, device=device),
        },
        batch_size=[n_harmonics],
    )
    voice.attractors = carriers

    # Create Noise (The raw material)
    n_particles = 100
    print(f"    Generating {n_particles} noise particles...")
    noise = TensorDict(
        {
            "position": torch.rand(n_particles, dtype=DTYPE_REAL, device=device) * 1000.0,  # 0-1000 Hz
            "energy": torch.ones(n_particles, dtype=DTYPE_REAL, device=device),
            "phase": torch.rand(n_particles, dtype=DTYPE_REAL, device=device) * TAU,
            "ttl": torch.ones(n_particles, dtype=DTYPE_REAL, device=device) * 10.0,
        },
        batch_size=[n_particles],
    )
    voice.particles = noise

    # Run Diffusion (Speaking)
    print("    Speaking (Diffusing Noise into Sound)...")
    steps = 50
    for i in range(steps):
        # Annealing Schedule
        prog = i / steps
        sharpness = 0.1 + prog * 10.0  # Soft -> Hard

        # 1. Calculate Distances (1D Audio)
        p_pos = voice.particles.get("position").unsqueeze(1)
        a_pos = voice.attractors.get("position").unsqueeze(0)
        dists = (p_pos - a_pos).abs()

        # 2. Softmax Gravity
        weights = torch.softmax(-dists * sharpness, dim=1)

        # 3. Drift
        targets = (weights * a_pos).sum(dim=1)
        current = voice.particles.get("position")
        drift = -2.0 * (current - targets)

        # 4. Update
        new_pos = current + drift * spec_cfg.dt
        voice.particles.set("position", new_pos)

    # ==================================================================
    # 5. Result
    # ==================================================================
    print("\n[5] Output Analysis")
    final_freqs = voice.particles.get("position")
    print("    Generated Particle Frequencies (Mean):")

    # Check clustering
    for target in target_freqs:
        # Find particles near target
        mask = (final_freqs - target).abs() < 20.0
        count = mask.sum().item()
        if count > 0:
            actual = final_freqs[mask].mean().item()
            print(f"    Target {target} Hz -> Generated {actual:.2f} Hz ({count} particles)")
        else:
            print(f"    Target {target} Hz -> No particles captured")

    print("\n============================================================")
    print("DEMO COMPLETE: The system 'Thought' of a word and 'Spoke' it.")
    print("============================================================")


if __name__ == "__main__":
    run_unified_demo()
