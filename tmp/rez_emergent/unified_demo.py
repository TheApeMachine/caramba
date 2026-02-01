
#!/usr/bin/env python3
"""
rez_emergent.unified_demo

End-to-end demo:
  SemanticManifold (thinking) -> BridgeManifold (transduction) -> SpectralManifold (speaking)

This file specifically addresses two problems called out in the critique of the *old*
codebase:

1) O(N*M) all-to-all distance loops in the physics core. 
   - The SpectralManifold now uses a local 1D horizon window for interactions
     (K≈sqrt(M) neighbors) when enabled.

2) A hard-coded semantic→audio dictionary. 
   - Replaced with BridgeManifold, trained by co-activation experience.

Run:
  python3 unified_demo.py
  python3 unified_demo.py --beefy
  python3 unified_demo.py --beefy-prob
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


import argparse
import json
import math
import wave
from dataclasses import asdict
from typing import Dict, List, Tuple

import torch

from physics import PhysicsConfig, SpectralManifold, DTYPE_REAL, TAU
from semantic import SemanticManifold
from bridge import BridgeManifold


def _semantic_vec_from_probs(probs: torch.Tensor, token_emb: torch.Tensor, eps: float) -> torch.Tensor:
    # probs: [V], token_emb: [V,D] => [D]
    vec = torch.mv(token_emb.t(), probs)
    return vec / vec.norm().clamp_min(eps)


def _make_environment_audio_mapping(
    device: torch.device,
    vocab_size: int,
    spec_freqs: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    Create a synthetic "world" mapping from token ids -> a sparse set of frequencies.

    This is not part of the model; it is the environment that provides
    co-activation experience so the BridgeManifold can learn an association.

    The mapping is randomized (not hand-designed) to avoid an "author-tuned" bridge.
    """
    B = int(spec_freqs.numel())
    # Emergent sparsity: each token activates K≈sqrt(B) bins.
    K = int(math.ceil(math.sqrt(float(B))))
    mapping: Dict[int, torch.Tensor] = {}
    for i in range(vocab_size):
        idx = torch.randperm(B, device=device)[:K]
        mapping[i] = spec_freqs.index_select(0, idx)
    return mapping


def run_unified_demo() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # -------------------------
    # 1) Semantic manifold
    # -------------------------
    vocab = ["The", "Cat", "Sat", "On", "Mat"]
    V = len(vocab)
    D = V

    sem_cfg = PhysicsConfig(dt=0.1)
    brain = SemanticManifold(sem_cfg, device=device, embed_dim=D, vocab_size=V, enable_event_horizon=False)

    # Train a tiny grammar from a short corpus: The Cat Sat On Mat
    # We treat each observed transition as a physical co-occurrence event.
    seq = list(range(V))

    # Train from repeated co-occurrence exposures (scale-derived).
    exposures = int(V * V * V)
    for _ in range(exposures):
        i = int(torch.randint(0, len(seq) - 1, (1,)).item())
        if i == 0:
            ctx_ids_train = [seq[0]]
            nxt = seq[1]
        else:
            ctx_ids_train = [seq[i - 1], seq[i]]
            nxt = seq[i + 1]
        brain.ingest_context(brain.attractors["position"][ctx_ids_train])
        brain.observe_next(nxt)

    # Input context: "The Cat"
    ctx_ids = [0, 1]
    brain.ingest_context(brain.attractors["position"][ctx_ids])
    sem_out = brain.output_state(vocab=vocab)

    if sem_out.probs is None or sem_out.token_index is None:
        raise RuntimeError("Semantic output missing probabilities or token index")

    print("============================================================")
    print("UNIFIED DEMO (rez_emergent)")
    print("============================================================")
    print(f"[Semantic] context: {' '.join(vocab[i] for i in ctx_ids)}")
    print(f"[Semantic] prediction: {sem_out.token} (id={sem_out.token_index})")
    if sem_out.meta is not None:
        print(f"[Semantic] meta: entropy={sem_out.meta.get('entropy'):.4f} confidence={sem_out.meta.get('confidence'):.3f} carriers={sem_out.meta.get('carriers')}")

    # -------------------------
    # 2) Bridge manifold
    # -------------------------
    # Spectral bank configuration (environment + bridge)
    spec_bins = 64
    # Choose an audible band (engineering choice, not tuned to the model).
    f_min = 180.0
    f_max = 900.0

    bridge_cfg = PhysicsConfig(dt=0.1)
    bridge = BridgeManifold(
        bridge_cfg,
        device=device,
        sem_dim=D,
        spec_bins=spec_bins,
        spec_min_hz=f_min,
        spec_max_hz=f_max,
        spec_embed_dim=None,  # derived from bins
        enable_event_horizon=False,
    )

    # Synthetic environment provides co-activation experiences for training.
    env_map = _make_environment_audio_mapping(device, V, bridge.spec_freqs)

    # Train bridge: show each token alongside its environment audio pattern.
    # Training exposure count derived from system scale (V * sqrt(B)).
    exposures = int(V * math.ceil(math.sqrt(float(spec_bins))))
    for _ in range(exposures):
        tok = int(torch.randint(0, V, (1,)).item())
        sem_vec = brain.attractors["position"][tok]
        audio_freqs = env_map[tok]
        bridge.observe(sem_vec, audio_freqs)

    # Inference: semantic prediction -> semantic vector -> bridge -> target freqs
    sem_vec = _semantic_vec_from_probs(sem_out.probs, brain.attractors["position"], eps=sem_cfg.eps)
    spec_vec = bridge.predict_vec(sem_vec)
    target_freqs, target_energies = bridge.decode_targets(spec_vec)

    print(f"[Bridge] carriers={bridge.metrics().carriers} nucleation_mass={bridge.metrics().nucleation_mass:.3f}")
    if target_freqs.numel() > 0:
        preview = ", ".join(f"{float(f):.1f}Hz" for f in target_freqs[: min(8, target_freqs.numel())])
        print(f"[Bridge] decoded targets: {preview}{' ...' if target_freqs.numel() > 8 else ''}")
    else:
        print("[Bridge] decoded targets: <none>")

    # -------------------------
    # 3) Spectral manifold (voice)
    # -------------------------
    voice_cfg = PhysicsConfig(dt=0.01, local_1d_horizon=True, max_particles=4096)
    voice = SpectralManifold(voice_cfg, device=device)

    # Noise seed (raw material)
    # Particle count derived from target set size and system cap.
    n_targets = int(target_freqs.numel())
    n_noise = min(voice_cfg.max_particles, max(1, n_targets * n_targets))
    voice.seed_noise(n_noise, f_min=float(target_freqs.min().item()) if n_targets else f_min, f_max=float(target_freqs.max().item()) if n_targets else f_max)

    # Set attractors from bridge outputs
    voice.set_targets(target_freqs, energies=target_energies)

    # Run physics
    steps = int(math.ceil(math.sqrt(float(n_noise + 1))))
    voice.step(steps)

    # Analyze particle clustering near each target
    freqs = voice.particles["position"]
    if freqs.numel() == 0 or target_freqs.numel() == 0:
        rendered: List[Tuple[float, float]] = []
    else:
        rendered = []
        # Tolerance derived from target spacing
        log_t = torch.log(target_freqs.clamp_min(1e-6))
        if log_t.numel() > 1:
            spacing = torch.mean(torch.abs(log_t[1:] - log_t[:-1])).item()
            tol = float(math.exp(spacing) - 1.0)  # relative tolerance
        else:
            tol = 0.05  # fallback for 1 target
        for f in target_freqs:
            mask = (freqs - f).abs() <= (tol * f)
            count = float(mask.sum().item())
            if count > 0:
                rendered.append((float(freqs[mask].mean().item()), count))

    # Render waveform from discovered partials
    out_dir = "tmp/rez_emergent"
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "unified_demo.wav")
    if rendered:
        max_f = max(f for f, _ in rendered)
        sample_rate = max(8000, int(max_f * 4.0))
        duration = steps * voice_cfg.dt
        n_samples = max(1, int(sample_rate * duration))
        t = torch.linspace(0.0, duration, n_samples, dtype=DTYPE_REAL, device=device)

        total = sum(c for _, c in rendered)
        wave_sum = torch.zeros_like(t)
        for f, c in rendered:
            amp = c / max(1.0, total)
            wave_sum = wave_sum + amp * torch.sin(2.0 * TAU * float(f) * t)

        peak = torch.max(torch.abs(wave_sum)).clamp_min(1e-8)
        wave_sum = wave_sum / peak
        audio = (wave_sum.cpu().numpy() * 32767.0).astype("<i2")

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        print(f"[Voice] wrote WAV: {wav_path} (partials={len(rendered)}, steps={steps})")
    else:
        print("[Voice] no partials captured; skipping WAV render")

    print("============================================================")


def run_beefy_ring() -> None:
    """
    Deterministic ring grammar: i -> (i+1)%V
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    V = 256
    D = 128
    vocab = [f"T{i}" for i in range(V)]

    cfg = PhysicsConfig(dt=0.05)
    brain = SemanticManifold(cfg, device=device, embed_dim=D, vocab_size=V, enable_event_horizon=True)

    # Train samples derived from scale: V * sqrt(V)
    train_samples = int(V * math.ceil(math.sqrt(float(V))))
    for _ in range(train_samples):
        i = int(torch.randint(0, V, (1,)).item())
        nxt = (i + 1) % V
        brain.ingest_context(brain.attractors["position"][i : i + 1])
        brain.observe_next(nxt)

    # Evaluate
    test = int(V)  # one per token
    correct = 0
    ent_sum = 0.0
    for i in range(test):
        brain.ingest_context(brain.attractors["position"][i : i + 1])
        out = brain.output_state(vocab=vocab)
        if out.token_index == (i + 1) % V:
            correct += 1
        ent_sum += float(out.meta.get("entropy", 0.0)) if out.meta else 0.0

    print("============================================================")
    print("BEEFY TEST: ring grammar")
    print("============================================================")
    print(f"vocab={V} embed_dim={D} train_samples={train_samples}")
    print(f"accuracy={correct / max(1, test):.3f} avg_entropy={ent_sum / max(1, test):.4f}")
    print("============================================================")


def run_beefy_prob() -> None:
    """
    Probabilistic grammar: each token has 3 outgoing edges sampled randomly.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    V = 256
    D = 128
    vocab = [f"T{i}" for i in range(V)]

    cfg = PhysicsConfig(dt=0.05)
    brain = SemanticManifold(cfg, device=device, embed_dim=D, vocab_size=V, enable_event_horizon=True)

    probs = torch.zeros((V, V), dtype=DTYPE_REAL, device=device)
    for i in range(V):
        choices = torch.randperm(V, device=device)[:3]
        w = torch.rand((3,), dtype=DTYPE_REAL, device=device)
        w = w / w.sum().clamp_min(1e-8)
        probs[i, choices] = w

    def sample_next(i: int) -> int:
        return int(torch.multinomial(probs[i], 1).item())

    train_samples = int(V * math.ceil(math.sqrt(float(V))))
    for _ in range(train_samples):
        i = int(torch.randint(0, V, (1,)).item())
        nxt = sample_next(i)
        brain.ingest_context(brain.attractors["position"][i : i + 1])
        brain.observe_next(nxt)

    # Evaluate KL on a random set
    test = int(V)
    kl_sum = 0.0
    for _ in range(test):
        i = int(torch.randint(0, V, (1,)).item())
        brain.ingest_context(brain.attractors["position"][i : i + 1])
        out = brain.output_state(vocab=vocab)
        pred = out.probs
        if pred is None:
            continue
        target = probs[i]
        kl = (target * (torch.log(target.clamp_min(1e-8)) - torch.log(pred.clamp_min(1e-8)))).sum()
        kl_sum += float(kl.item())

    print("============================================================")
    print("BEEFY TEST: probabilistic grammar")
    print("============================================================")
    print(f"vocab={V} embed_dim={D} train_samples={train_samples}")
    print(f"avg_KL(target||pred)={kl_sum / max(1, test):.4f}")
    print("============================================================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beefy", action="store_true", help="Run deterministic ring grammar test")
    ap.add_argument("--beefy-prob", action="store_true", help="Run probabilistic grammar test")
    args = ap.parse_args()

    if args.beefy:
        run_beefy_ring()
    elif args.beefy_prob:
        run_beefy_prob()
    else:
        run_unified_demo()