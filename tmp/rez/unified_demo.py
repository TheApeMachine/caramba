#!/usr/bin/env python3
"""
The Unified Manifold: Text-to-Physics-to-Audio Pipeline

This script demonstrates the complete "System 2" loop:
1. Semantic Physics: "Thinking" (Energy flows through grammar)
2. The Bridge: Concept -> Frequency Projection
3. Spectral Physics: "Speaking" (Diffusion generates audio)
"""

import argparse
import json
import math
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
    brain = SemanticManifold(sem_cfg, device, embed_dim, vocab_size, num_modes=3)

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
    if probs is None or next_token_idx is None:
        raise RuntimeError("Semantic output missing probabilities or token index")

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

    if final_freqs is None or final_freqs.numel() == 0:
        print("    No particles available for clustering")
        rendered_partials = []
    else:
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


def run_beefy_test() -> None:
    print("============================================================")
    print("THERMODYNAMIC MANIFOLD: Beefy Test")
    print("============================================================")
    torch.manual_seed(0)
    device = torch.device("cpu")

    vocab_size = 64
    vocab = [f"T{i}" for i in range(vocab_size)]
    embed_dim = vocab_size
    sem_cfg = SemanticPhysicsConfig(dt=0.1)
    brain = SemanticManifold(sem_cfg, device, embed_dim, vocab_size, num_modes=4)

    # Deterministic next-token grammar (ring)
    next_map = {i: (i + 1) % vocab_size for i in range(vocab_size)}

    # Training: expose many short sequences to build bond energy
    train_sequences = 500
    seq_len = 8
    for _ in range(train_sequences):
        start = int(torch.randint(0, vocab_size, (1,)).item())
        seq = [(start + i) % vocab_size for i in range(seq_len)]
        embeddings = brain.attractors.get("position")[seq]
        brain.ingest_context(embeddings)
        exc = brain.attractors.get("excitation")
        exc[seq[-1]] = 1.0
        brain.attractors.set("excitation", exc)
        for _ in range(seq_len):
            brain.step_grammar()

    # Evaluation: predict next token for short contexts
    test_cases = 200
    correct = 0
    steps_used_total = 0
    dominance_total = 0.0
    entropy_total = 0.0
    entropy_delta_total = 0.0
    mode_entropy_total = 0.0
    confidence_total = 0.0
    mode_entropy_norm_total = 0.0
    heat_total = 0.0
    heat_var_total = 0.0
    dbg_totals: dict[str, float] = {}
    dbg_counts: dict[str, int] = {}
    max_steps = brain.vocab_size + seq_len
    for _ in range(test_cases):
        start = int(torch.randint(0, vocab_size, (1,)).item())
        context = [start, next_map[start]]
        expected = next_map[context[-1]]
        embeddings = brain.attractors.get("position")[context]
        brain.ingest_context(embeddings)
        exc = brain.attractors.get("excitation")
        exc[context[-1]] = 1.0
        brain.attractors.set("excitation", exc)

        steps_used = 0
        ent_start = float(brain.entropy().item())
        while steps_used < max_steps:
            brain.step_grammar()
            steps_used += 1
            if hasattr(brain, "last_debug"):
                for key, value in brain.last_debug.items():
                    dbg_totals[key] = dbg_totals.get(key, 0.0) + float(value)
                    dbg_counts[key] = dbg_counts.get(key, 0) + 1
            if brain.thinking_complete():
                break
        steps_used_total += steps_used
        dominance_total += brain.dominance_metrics()["dominance"]
        ent_end = float(brain.entropy().item())
        entropy_total += ent_end
        entropy_delta_total += ent_start - ent_end
        if brain.last_mode_weights is not None:
            m = brain.last_mode_weights
            mode_entropy_total += float(-(m * torch.log(m + 1e-8)).sum().item())
        confidence_total += brain.thinking_confidence()
        if brain.num_modes > 1 and brain.last_mode_entropy is not None:
            mode_entropy_norm_total += brain.last_mode_entropy / max(1e-8, math.log(float(brain.num_modes)))
        if "heat" in brain.attractors.keys():
            h = brain.attractors.get("heat")
            heat_total += float(h.mean().item()) if h.numel() > 0 else 0.0
            heat_var_total += float(h.var(unbiased=False).item()) if h.numel() > 0 else 0.0
        out = brain.output_state(vocab=vocab)
        if out.token_index == expected:
            correct += 1

    accuracy = correct / max(1, test_cases)
    avg_steps = steps_used_total / max(1, test_cases)
    avg_dominance = dominance_total / max(1, test_cases)
    avg_entropy = entropy_total / max(1, test_cases)
    avg_entropy_delta = entropy_delta_total / max(1, test_cases)
    avg_mode_entropy = mode_entropy_total / max(1, test_cases)
    avg_confidence = confidence_total / max(1, test_cases)
    avg_mode_entropy_norm = mode_entropy_norm_total / max(1, test_cases)
    avg_heat = heat_total / max(1, test_cases)
    avg_heat_var = heat_var_total / max(1, test_cases)
    dbg_avgs = {key: (dbg_totals[key] / max(1, dbg_counts.get(key, 1))) for key in dbg_totals.keys()}

    print(f"    Vocab size: {vocab_size}")
    print(f"    Train sequences: {train_sequences} (len={seq_len})")
    print(f"    Test cases: {test_cases}")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    Avg thinking steps: {avg_steps:.2f}")
    print(f"    Avg grammar dominance: {avg_dominance:.3f}")
    print(f"    Avg confidence: {avg_confidence:.3f}")
    print(f"    Avg entropy: {avg_entropy:.4f}")
    print(f"    Avg entropy drop: {avg_entropy_delta:.4f}")
    print(f"    Avg heat: {avg_heat:.4f}")
    print(f"    Avg heat var: {avg_heat_var:.6f}")
    if dbg_avgs:
        print("    Debug means:")
        for key in sorted(dbg_avgs.keys()):
            print(f"      {key}: {dbg_avgs[key]:.6f}")
    if brain.num_modes > 1:
        print(f"    Avg mode entropy: {avg_mode_entropy:.4f}")
        print(f"    Avg mode entropy (norm): {avg_mode_entropy_norm:.4f}")

    report = {
        "vocab_size": vocab_size,
        "train_sequences": train_sequences,
        "sequence_length": seq_len,
        "test_cases": test_cases,
        "accuracy": accuracy,
        "avg_steps": avg_steps,
        "avg_dominance": avg_dominance,
        "avg_confidence": avg_confidence,
        "avg_entropy": avg_entropy,
        "avg_entropy_drop": avg_entropy_delta,
        "avg_mode_entropy": avg_mode_entropy,
        "avg_mode_entropy_norm": avg_mode_entropy_norm,
        "avg_heat": avg_heat,
        "avg_heat_var": avg_heat_var,
        "debug_means": dbg_avgs,
        "energy_metrics": brain.energy_metrics(),
    }
    artifacts_dir = "tmp/rez/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "beefy_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"    Wrote report: {report_path}")
    print("============================================================")


def run_probabilistic_beefy_test() -> None:
    print("============================================================")
    print("THERMODYNAMIC MANIFOLD: Probabilistic Beefy Test")
    print("============================================================")
    torch.manual_seed(0)
    device = torch.device("cpu")

    vocab_size = 64
    vocab = [f"T{i}" for i in range(vocab_size)]
    embed_dim = vocab_size
    sem_cfg = SemanticPhysicsConfig(dt=0.1)
    brain = SemanticManifold(sem_cfg, device, embed_dim, vocab_size, num_modes=4)

    # Build probabilistic grammar: 3 outgoing edges per token
    probs = torch.zeros(vocab_size, vocab_size, dtype=DTYPE_REAL, device=device)
    for i in range(vocab_size):
        choices = torch.randperm(vocab_size)[:3]
        weights = torch.rand(3, dtype=DTYPE_REAL, device=device)
        weights = weights / (weights.sum() + 1e-8)
        probs[i, choices] = weights

    def sample_next(token: int) -> int:
        return int(torch.multinomial(probs[token], 1).item())

    # Training: sample sequences from the probabilistic grammar
    train_sequences = 800
    seq_len = 8
    for _ in range(train_sequences):
        start = int(torch.randint(0, vocab_size, (1,)).item())
        seq = [start]
        for _ in range(seq_len - 1):
            seq.append(sample_next(seq[-1]))
        embeddings = brain.attractors.get("position")[seq]
        brain.ingest_context(embeddings)
        exc = brain.attractors.get("excitation")
        exc[seq[-1]] = 1.0
        brain.attractors.set("excitation", exc)
        for _ in range(seq_len):
            brain.step_grammar()

    # Evaluation: compare predicted distribution to ground-truth
    test_cases = 200
    kl_total = 0.0
    correct = 0
    steps_used_total = 0
    dominance_total = 0.0
    entropy_total = 0.0
    entropy_delta_total = 0.0
    mode_entropy_total = 0.0
    confidence_total = 0.0
    mode_entropy_norm_total = 0.0
    heat_total = 0.0
    heat_var_total = 0.0
    dbg_totals: dict[str, float] = {}
    dbg_counts: dict[str, int] = {}
    max_steps = brain.vocab_size + seq_len
    for _ in range(test_cases):
        start = int(torch.randint(0, vocab_size, (1,)).item())
        next_tok = sample_next(start)
        context = [start, next_tok]
        embeddings = brain.attractors.get("position")[context]
        brain.ingest_context(embeddings)
        exc = brain.attractors.get("excitation")
        exc[context[-1]] = 1.0
        brain.attractors.set("excitation", exc)

        steps_used = 0
        ent_start = float(brain.entropy().item())
        while steps_used < max_steps:
            brain.step_grammar()
            steps_used += 1
            if hasattr(brain, "last_debug"):
                for key, value in brain.last_debug.items():
                    dbg_totals[key] = dbg_totals.get(key, 0.0) + float(value)
                    dbg_counts[key] = dbg_counts.get(key, 0) + 1
            if brain.thinking_complete():
                break
        steps_used_total += steps_used
        dominance_total += brain.dominance_metrics()["dominance"]
        ent_end = float(brain.entropy().item())
        entropy_total += ent_end
        entropy_delta_total += ent_start - ent_end
        if brain.last_mode_weights is not None:
            m = brain.last_mode_weights
            mode_entropy_total += float(-(m * torch.log(m + 1e-8)).sum().item())
        confidence_total += brain.thinking_confidence()
        if brain.num_modes > 1 and brain.last_mode_entropy is not None:
            mode_entropy_norm_total += brain.last_mode_entropy / max(1e-8, math.log(float(brain.num_modes)))
        if "heat" in brain.attractors.keys():
            h = brain.attractors.get("heat")
            heat_total += float(h.mean().item()) if h.numel() > 0 else 0.0
            heat_var_total += float(h.var(unbiased=False).item()) if h.numel() > 0 else 0.0

        out = brain.output_state(vocab=vocab)
        pred = out.probs
        if pred is None:
            continue
        target = probs[context[-1]]
        kl = (target * (torch.log(target + 1e-8) - torch.log(pred + 1e-8))).sum()
        kl_total += float(kl.item())

        expected = int(torch.argmax(target).item())
        if out.token_index == expected:
            correct += 1

    accuracy = correct / max(1, test_cases)
    avg_steps = steps_used_total / max(1, test_cases)
    avg_dominance = dominance_total / max(1, test_cases)
    avg_kl = kl_total / max(1, test_cases)
    avg_entropy = entropy_total / max(1, test_cases)
    avg_entropy_delta = entropy_delta_total / max(1, test_cases)
    avg_mode_entropy = mode_entropy_total / max(1, test_cases)
    avg_confidence = confidence_total / max(1, test_cases)
    avg_mode_entropy_norm = mode_entropy_norm_total / max(1, test_cases)
    avg_heat = heat_total / max(1, test_cases)
    avg_heat_var = heat_var_total / max(1, test_cases)
    dbg_avgs = {key: (dbg_totals[key] / max(1, dbg_counts.get(key, 1))) for key in dbg_totals.keys()}

    print(f"    Vocab size: {vocab_size}")
    print(f"    Train sequences: {train_sequences} (len={seq_len})")
    print(f"    Test cases: {test_cases}")
    print(f"    Top-1 Accuracy: {accuracy:.3f}")
    print(f"    Avg KL (target || pred): {avg_kl:.4f}")
    print(f"    Avg thinking steps: {avg_steps:.2f}")
    print(f"    Avg grammar dominance: {avg_dominance:.3f}")
    print(f"    Avg confidence: {avg_confidence:.3f}")
    print(f"    Avg entropy: {avg_entropy:.4f}")
    print(f"    Avg entropy drop: {avg_entropy_delta:.4f}")
    print(f"    Avg heat: {avg_heat:.4f}")
    print(f"    Avg heat var: {avg_heat_var:.6f}")
    if dbg_avgs:
        print("    Debug means:")
        for key in sorted(dbg_avgs.keys()):
            print(f"      {key}: {dbg_avgs[key]:.6f}")
    if brain.num_modes > 1:
        print(f"    Avg mode entropy: {avg_mode_entropy:.4f}")
        print(f"    Avg mode entropy (norm): {avg_mode_entropy_norm:.4f}")

    report = {
        "vocab_size": vocab_size,
        "train_sequences": train_sequences,
        "sequence_length": seq_len,
        "test_cases": test_cases,
        "top1_accuracy": accuracy,
        "avg_kl": avg_kl,
        "avg_steps": avg_steps,
        "avg_dominance": avg_dominance,
        "avg_confidence": avg_confidence,
        "avg_entropy": avg_entropy,
        "avg_entropy_drop": avg_entropy_delta,
        "avg_mode_entropy": avg_mode_entropy,
        "avg_mode_entropy_norm": avg_mode_entropy_norm,
        "avg_heat": avg_heat,
        "avg_heat_var": avg_heat_var,
        "debug_means": dbg_avgs,
        "energy_metrics": brain.energy_metrics(),
    }
    artifacts_dir = "tmp/rez/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "beefy_report_prob.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"    Wrote report: {report_path}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beefy", action="store_true", help="Run a larger-scale grammar test")
    parser.add_argument("--beefy-prob", action="store_true", help="Run a probabilistic grammar test")
    args = parser.parse_args()
    if args.beefy:
        run_beefy_test()
    elif args.beefy_prob:
        run_probabilistic_beefy_test()
    else:
        run_unified_demo()
