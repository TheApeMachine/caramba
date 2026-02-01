#!/usr/bin/env python3
"""
The Unified Manifold: Text-to-Physics-to-Audio Pipeline

This script demonstrates the complete "System 2" loop:
1. Semantic Physics: "Thinking" (Energy flows through grammar)
2. The Bridge: Concept -> Frequency Projection
3. Spectral Physics: "Speaking" (Diffusion generates audio)
"""

import os
import torch
import wave
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
    embed_dim = vocab_size

    # Physics Config: semantic integration step
    sem_cfg = SemanticPhysicsConfig(dt=0.1)
    brain = SemanticManifold(sem_cfg, device, embed_dim, vocab_size)

    print("    Grammar: emergent from attractor geometry (no explicit seeding)")

    # Define "Musical Concepts" (The Bridge)
    # We map each word's embedding to a specific Chord (Target Frequencies)
    # In this setup, this stands in for the learned 'bridge_matrix'
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
    print("\n[2] Learning Grammar from Context...")
    # Provide a short sequence so grammar bonds can emerge from context order
    full_embeddings = brain.attractors.get("position")
    brain.ingest_context(full_embeddings)
    learn_steps = int(brain.particles.shape[0])
    for _ in range(learn_steps):
        brain.step_grammar()

    print("\n[3] Input Context: 'The Cat'")
    # Ingest "The Cat"
    embeddings = brain.attractors.get("position")[[0, 1]]
    brain.ingest_context(embeddings)

    # Manually excite "Cat" (most recent)
    exc = brain.attractors.get("excitation")
    exc[1] = 1.0
    brain.attractors.set("excitation", exc)

    print("    Thinking (Running Thermodynamic Grammar)...")
    # No tuning: let the system decide when to stop (entropy settles)
    steps_used = 0
    entropy_history = []
    dominance_history = []
    max_steps = brain.vocab_size + int(brain.particles.shape[0])
    while steps_used < max_steps:
        brain.step_grammar()
        steps_used += 1
        entropy = brain.entropy()
        entropy_history.append(float(entropy.item()))
        dominance_history.append(brain.dominance_metrics()["dominance"])
        if brain.thinking_complete():
            break

    semantic_out = brain.output_state(vocab=vocab)
    probs = semantic_out.probs
    next_token_idx = semantic_out.token_index
    predicted_word = semantic_out.token

    print(f"    Brain Prediction: '{predicted_word}' (Confidence: {probs[next_token_idx]:.2f})")
    print(f"    (Ran {steps_used} grammar step(s); no manual tuning)")
    if entropy_history:
        formatted = ", ".join(f"{e:.4f}" for e in entropy_history)
        print(f"    Entropy curve: [{formatted}]")
    if dominance_history:
        formatted = ", ".join(f"{d:.3f}" for d in dominance_history)
        print(f"    Grammar dominance: [{formatted}]")

    # ==================================================================
    # 3. The Bridge (Concept -> Physics)
    # ==================================================================
    print("\n[4] Bridging Concept to Physics...")

    target_freqs = word_to_audio[next_token_idx]
    print(f"    Concept '{predicted_word}' maps to Frequencies: {target_freqs} Hz")

    # ==================================================================
    # 4. The "Speaking" Phase (Spectral Manifold)
    # ==================================================================
    print("\n[5] Initializing The Voice (Spectral Physics)...")

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
    print("\n[6] Output Analysis")
    audio_out = voice.output_state()
    final_freqs = audio_out.audio_particles
    print("    Generated Particle Frequencies (Mean):")

    # Check clustering
    rendered_partials: list[tuple[float, float]] = []
    for target in target_freqs:
        # Find particles near target
        mask = (final_freqs - target).abs() < 20.0
        count = mask.sum().item()
        if count > 0:
            actual = final_freqs[mask].mean().item()
            print(f"    Target {target} Hz -> Generated {actual:.2f} Hz ({count} particles)")
            rendered_partials.append((actual, count))
        else:
            print(f"    Target {target} Hz -> No particles captured")

    # Render a simple waveform from the generated partials
    if rendered_partials:
        max_freq = max(freq for freq, _ in rendered_partials)
        sample_rate = max(8000, int(max_freq * 4.0))
        duration = steps * spec_cfg.dt
        n_samples = max(1, int(sample_rate * duration))
        t = torch.linspace(0.0, duration, n_samples, dtype=DTYPE_REAL, device=device)
        total_count = sum(count for _, count in rendered_partials)
        wave_sum = torch.zeros_like(t)
        for freq, count in rendered_partials:
            amp = count / max(1.0, total_count)
            wave_sum = wave_sum + amp * torch.sin(2.0 * TAU * freq * t)
        peak = torch.max(torch.abs(wave_sum)) + 1e-8
        wave_sum = wave_sum / peak
        audio = (wave_sum.cpu().numpy() * 32767.0).astype("<i2")
        wav_path = "tmp/rez/unified_demo.wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        print(f"    Wrote WAV: {wav_path} ({sample_rate} Hz, {duration:.2f}s)")

    print("\n============================================================")
    print("DEMO COMPLETE: The system 'Thought' of a word and 'Spoke' it.")
    print("============================================================")


if __name__ == "__main__":
    run_unified_demo()
