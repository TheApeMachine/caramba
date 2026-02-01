"""
Generate all paper artifacts for tmp/rez/paper/main_thermo.tex in one command.

Usage:
  python3 tmp/rez/paper_artifacts.py

This script is headless (Agg backend) and writes outputs to:
  tmp/rez/artifacts/
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from physics import PhysicsConfig, SpectralManifold, TAU
from semantic import SemanticManifold


REZ_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = REZ_DIR / "artifacts"

# Avoid noisy warnings about unwritable cache directories (common in sandboxes).
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ARTIFACTS_DIR / ".cache"))

# Headless plotting (backend must be set before importing pyplot)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _run_semantic_demo():
    device = torch.device("cpu")
    vocab = ["The", "Cat", "Sat", "On", "Mat"]
    vocab_size = len(vocab)
    embed_dim = vocab_size

    brain = SemanticManifold(PhysicsConfig(dt=0.1), device, embed_dim, vocab_size)

    # Learn grammar from full context order
    full_embeddings = brain.attractors.get("position")
    brain.ingest_context(full_embeddings)
    for _ in range(int(brain.particles.shape[0])):
        brain.step_grammar()

    # Predict after "The Cat"
    embeddings = brain.attractors.get("position")[[0, 1]]
    brain.ingest_context(embeddings)
    exc = brain.attractors.get("excitation")
    exc[1] = 1.0
    brain.attractors.set("excitation", exc)

    entropy_hist = []
    dominance_hist = []
    steps_used = 0
    max_steps = brain.vocab_size + int(brain.particles.shape[0])
    while steps_used < max_steps:
        brain.step_grammar()
        steps_used += 1
        entropy_hist.append(float(brain.entropy().item()))
        dominance_hist.append(brain.dominance_metrics()["dominance"])
        if brain.thinking_complete():
            break

    out = brain.output_state(vocab=vocab)
    return {
        "vocab": vocab,
        "out": out,
        "steps_used": steps_used,
        "entropy": entropy_hist,
        "dominance": dominance_hist,
        "device": device,
    }


def _run_spectral_demo(target_freqs: list[float], device: torch.device):
    spec_cfg = PhysicsConfig(dt=0.01)
    voice = SpectralManifold(spec_cfg, device)

    n_harmonics = len(target_freqs)
    carriers = {
        "id": torch.arange(n_harmonics, device=device),
        "position": torch.tensor(target_freqs, dtype=torch.float32, device=device),
        "phase": torch.zeros(n_harmonics, dtype=torch.float32, device=device),
        "gate_width": torch.ones(n_harmonics, dtype=torch.float32, device=device) * 0.5,
        "heat": torch.zeros(n_harmonics, dtype=torch.float32, device=device),
        "energy": torch.ones(n_harmonics, dtype=torch.float32, device=device),
    }
    from tensordict import TensorDict

    voice.attractors = TensorDict(carriers, batch_size=[n_harmonics])

    n_particles = 100
    noise = TensorDict(
        {
            "position": torch.rand(n_particles, dtype=torch.float32, device=device) * 1000.0,
            "energy": torch.ones(n_particles, dtype=torch.float32, device=device),
            "phase": torch.rand(n_particles, dtype=torch.float32, device=device) * TAU,
            "ttl": torch.ones(n_particles, dtype=torch.float32, device=device) * 10.0,
        },
        batch_size=[n_particles],
    )
    voice.particles = noise

    steps = 50
    for i in range(steps):
        prog = i / steps
        sharpness = 0.1 + prog * 10.0
        p_pos = voice.particles.get("position").unsqueeze(1)
        a_pos = voice.attractors.get("position").unsqueeze(0)
        dists = (p_pos - a_pos).abs()
        weights = torch.softmax(-dists * sharpness, dim=1)
        targets = (weights * a_pos).sum(dim=1)
        current = voice.particles.get("position")
        drift = -2.0 * (current - targets)
        new_pos = current + drift * spec_cfg.dt
        voice.particles.set("position", new_pos)

    return voice.particles.get("position").detach().cpu().numpy()


def _summarize_targets(target_freqs: list[float], final_freqs: np.ndarray):
    rows = []
    for target in target_freqs:
        mask = np.abs(final_freqs - target) < 20.0
        count = int(mask.sum())
        if count > 0:
            actual = float(final_freqs[mask].mean())
        else:
            actual = float("nan")
        rows.append({"target": float(target), "actual": actual, "count": count})
    return rows


def _render_waveform(rendered_partials: list[tuple[float, float]], duration: float):
    if not rendered_partials:
        return None, None, None
    max_freq = max(freq for freq, _ in rendered_partials)
    sample_rate = max(8000, int(max_freq * 4.0))
    n_samples = max(1, int(sample_rate * duration))
    t = torch.linspace(0.0, duration, n_samples, dtype=torch.float32)
    total_count = sum(count for _, count in rendered_partials)
    wave_sum = torch.zeros_like(t)
    for freq, count in rendered_partials:
        amp = count / max(1.0, total_count)
        wave_sum = wave_sum + amp * torch.sin(2.0 * TAU * freq * t)
    peak = torch.max(torch.abs(wave_sum)) + 1e-8
    wave_sum = wave_sum / peak
    audio = (wave_sum.numpy() * 32767.0).astype("<i2")
    return audio, sample_rate, duration


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    semantic = _run_semantic_demo()
    out = semantic["out"]
    vocab = semantic["vocab"]
    device = semantic["device"]

    word_to_audio = {
        0: [220.0, 440.0],
        1: [300.0, 350.0, 400.0],
        2: [261.6, 329.6, 392.0],
        3: [329.6, 415.3, 493.9],
        4: [261.6, 261.6, 261.6],
    }
    target_freqs = word_to_audio[int(out.token_index)]
    final_freqs = _run_spectral_demo(target_freqs, device)
    summary_rows = _summarize_targets(target_freqs, final_freqs)
    rendered_partials = [(row["actual"], row["count"]) for row in summary_rows if not np.isnan(row["actual"])]

    # --- JSON summary ---
    results = {
        "token": out.token,
        "token_index": out.token_index,
        "confidence": float(out.probs[out.token_index]),
        "steps_used": semantic["steps_used"],
        "entropy": semantic["entropy"],
        "dominance": semantic["dominance"],
        "target_freqs": target_freqs,
        "summary_rows": summary_rows,
    }
    (ARTIFACTS_DIR / "thermo_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # --- Entropy + dominance plot ---
    fig = plt.figure(figsize=(7.6, 3.6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    xs = np.arange(1, len(semantic["entropy"]) + 1)
    ax.plot(xs, semantic["entropy"], linewidth=1.6, label="entropy")
    ax.plot(xs, semantic["dominance"], linewidth=1.6, label="grammar dominance")
    ax.set_xlabel("thinking step")
    ax.set_title("Entropy and grammar dominance")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(ARTIFACTS_DIR / "thermo_entropy_dominance.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # --- Audio histogram plot ---
    fig = plt.figure(figsize=(7.6, 3.6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(final_freqs, bins=30, alpha=0.85, color="#4c78a8")
    for target in target_freqs:
        ax.axvline(target, color="#f58518", linewidth=1.5)
    ax.set_title("Generated particle frequencies")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.4)
    fig.savefig(ARTIFACTS_DIR / "thermo_audio_hist.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # --- Render waveform + spectrogram ---
    audio, sample_rate, duration = _render_waveform(rendered_partials, duration=0.5)
    if audio is not None and sample_rate is not None:
        wav_path = ARTIFACTS_DIR / "thermo_demo.wav"
        import wave

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        fig = plt.figure(figsize=(7.6, 3.6), facecolor="white")
        ax = fig.add_subplot(1, 1, 1)
        ax.specgram(audio.astype(float) / 32768.0, NFFT=256, Fs=sample_rate, noverlap=128, cmap="magma")
        ax.set_title("Synthesized waveform spectrogram")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        fig.tight_layout(pad=0.4)
        fig.savefig(ARTIFACTS_DIR / "thermo_spectrogram.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # --- LaTeX snippet ---
    lines: list[str] = []
    lines.append(r"% Generated by tmp/rez/paper_artifacts.py.")
    lines.append(r"\paragraph{Unified demo.}")
    lines.append(
        rf"The semantic manifold predicts \textbf{{{out.token}}} with confidence {float(out.probs[out.token_index]):.2f} "
        rf"after {semantic['steps_used']} thinking step(s). "
        rf"Mean entropy is {float(np.mean(semantic['entropy'])):.3f}, final grammar dominance is "
        rf"{float(semantic['dominance'][-1]) if semantic['dominance'] else 0.0:.3f}."
    )
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Generated spectral clusters vs. targets.}")
    lines.append(r"\begin{tabular}{@{}rrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Target (Hz) & Generated (Hz) & Particles \\")
    lines.append(r"\midrule")
    for row in summary_rows:
        generated = f"{row['actual']:.2f}" if not np.isnan(row["actual"]) else "--"
        lines.append(rf"{row['target']:.1f} & {generated} & {row['count']} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    (ARTIFACTS_DIR / "thermo_summary_autogen.tex").write_text("\n".join(lines) + "\n")

    end = time.perf_counter()
    print(f"[ok] wrote thermodynamic artifacts to {ARTIFACTS_DIR} (wall_s={end-start:.3f})")


if __name__ == "__main__":
    main()

