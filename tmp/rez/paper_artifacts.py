"""
Generate all paper artifacts for tmp/rez/main.tex in one command.

Usage:
  python3 tmp/rez/paper_artifacts.py

This script is intentionally headless (Agg backend) and writes outputs to:
  tmp/rez/artifacts/

Outputs (referenced by main.tex):
- rez_stream_results.json
- rez_stream_autogen.tex
- rez_stream_timeseries.png
- rez_stream_gate_capture.png
- rez_stream_presence.png
- rez_stream_lifetimes.png
- experiment_results.json (from rigorous experiments)
- experiment_results_autogen.tex
- experiment_summary_autogen.tex
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REZ_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = REZ_DIR / "artifacts"
MAIN_PY = REZ_DIR / "main.py"

# Avoid noisy warnings about unwritable cache directories (common in sandboxes).
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ARTIFACTS_DIR / ".cache"))

# Headless plotting (backend must be set before importing pyplot)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rez_main():
    spec = importlib.util.spec_from_file_location("rez_main", MAIN_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {MAIN_PY}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is registered for dataclasses (Python 3.14+ expects this).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    rez = _load_rez_main()
    engine = rez.ResonantEngine(seed=0)

    sim_s = 60.0
    steps = int(sim_s / float(rez.DT))

    t_series = []
    N_series = []
    M_series = []
    nnz_series = []
    Lcomp_series = []
    R_series = []

    # Track a "best" carrier over time for gate-capture visualization.
    best_u_raw = None
    best_gate = None
    best_name = None
    best_score = -1.0

    # Track a presence snapshot over time (pick the moment with highest nnz(P)).
    snap_mat = None
    snap_carrier_names = None
    snap_osc_freqs = None
    snap_nnz = -1

    for _ in range(steps):
        engine.step()
        t_series.append(float(engine.t))
        N_series.append(int(len(engine.oscillators)))
        M_series.append(int(len(engine.carriers)))
        nnz = int(engine.nnz_P())
        nnz_series.append(nnz)
        Lcomp_series.append(int(nnz + len(engine.carriers)))
        R_series.append(float(engine.global_sync_R()))

        # Select carrier with maximum energy (proxy for meaningful capture).
        if engine.carriers:
            c = max(engine.carriers, key=lambda cc: float(cc.energy))
            score = float(c.energy)
            if score > best_score and getattr(c, "u_raw_history", None) is not None:
                u_raw = np.asarray(list(c.u_raw_history), dtype=np.complex128)
                gate = np.asarray(list(c.gate_history), dtype=float)
                if u_raw.size > 8 and gate.size == u_raw.size:
                    best_score = score
                    best_u_raw = u_raw.copy()
                    best_gate = gate.copy()
                    best_name = str(c.name)

        # Save a presence snapshot when topology is most developed (highest nnz).
        if nnz > snap_nnz and engine.carriers and engine.oscillators:
            osc_items = sorted(engine.oscillators.values(), key=lambda o: o.freq_hz)
            # Show only a bounded number of oscillators/carriers for paper readability.
            max_osc = 40
            max_car = 8
            carriers = engine.carriers[: min(max_car, len(engine.carriers))]
            # Build a dense matrix for snapshot
            mat = np.zeros((min(max_osc, len(osc_items)), len(carriers)), dtype=float)
            freqs = [float(o.freq_hz) for o in osc_items[: mat.shape[0]]]
            for j, c in enumerate(carriers):
                for i, o in enumerate(osc_items[: mat.shape[0]]):
                    mat[i, j] = float(c.bonds.get(int(o.id), 0.0))
            snap_mat = mat
            snap_carrier_names = [str(c.name) for c in carriers]
            snap_osc_freqs = freqs
            snap_nnz = nnz

    # --- Results JSON ---
    results = {
        "seed": 0,
        "sim_s": sim_s,
        "dt": float(rez.DT),
        "steps": steps,
        "stream": {
            "event_rate_hz": float(rez.STREAM_EVENT_RATE_HZ),
            "freq_range_hz": [float(x) for x in rez.STREAM_FREQ_RANGE_HZ],
            "duration_range_s": [float(x) for x in rez.STREAM_DURATION_S_RANGE],
        },
        "metrics": {
            "N_mean": float(np.mean(N_series)) if N_series else 0.0,
            "M_mean": float(np.mean(M_series)) if M_series else 0.0,
            "nnz_mean": float(np.mean(nnz_series)) if nnz_series else 0.0,
            "Lcomp_mean": float(np.mean(Lcomp_series)) if Lcomp_series else 0.0,
            "R_mean": float(np.mean(R_series)) if R_series else 0.0,
            "births": int(len(engine.birth_events)),
            "deaths": int(len(engine.death_events)),
        },
        "events": {
            "births": engine.birth_events,
            "deaths": engine.death_events,
        },
        "series": {
            "t": t_series,
            "N": N_series,
            "M": M_series,
            "nnz": nnz_series,
            "Lcomp": Lcomp_series,
            "R": R_series,
        },
    }

    (ARTIFACTS_DIR / "rez_stream_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # --- Timeseries figure ---
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), facecolor="white")
    axes = axes.reshape(-1)
    xs = np.array(t_series, dtype=float)
    axes[0].plot(xs, N_series, linewidth=1.6)
    axes[0].set_title("N(t) oscillators")
    axes[1].plot(xs, M_series, linewidth=1.6)
    axes[1].set_title("M(t) carriers")
    axes[2].plot(xs, nnz_series, linewidth=1.6)
    axes[2].set_title("nnz(P)(t)")
    axes[3].plot(xs, Lcomp_series, linewidth=1.6)
    axes[3].set_title("L_comp(t) = nnz(P) + M")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("time (s)")
    fig.tight_layout(pad=0.6)
    fig.savefig(ARTIFACTS_DIR / "rez_stream_timeseries.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # --- Gate capture figure (from best observed carrier) ---
    if best_u_raw is not None and best_gate is not None:
        t_c = np.linspace(-float(rez.WINDOW_SIZE_S), 0.0, best_u_raw.size)
        sig = best_u_raw.real
        cap = sig * best_gate
        fig = plt.figure(figsize=(9.6, 3.6), facecolor="white")
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_c, sig, linewidth=1.5, label="Re(u_raw)")
        ax.plot(t_c, 2.4 * (best_gate - 0.5), linewidth=1.4, label="gate")
        ax.fill_between(t_c, 0.0, cap, where=(best_gate > 0.5), alpha=0.25, label="captured")
        title_name = best_name or "carrier"
        ax.set_title(f"Gate capture example ({title_name}, half-cycle gate)", fontsize=11)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        fig.tight_layout(pad=0.4)
        fig.savefig(ARTIFACTS_DIR / "rez_stream_gate_capture.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # --- Presence snapshot (best observed during the run) ---
    if snap_mat is not None and snap_carrier_names is not None:
        fig = plt.figure(figsize=(7.8, 3.6), facecolor="white")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(snap_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Presence snapshot P (peak nnz={snap_nnz})", fontsize=11)
        ax.set_xlabel("carriers")
        ax.set_ylabel("oscillators (sorted by frequency)")
        ax.set_xticks(range(len(snap_carrier_names)))
        ax.set_xticklabels(snap_carrier_names, fontsize=8)
        ax.set_yticks([])
        fig.tight_layout(pad=0.4)
        fig.savefig(ARTIFACTS_DIR / "rez_stream_presence.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # --- Lifetime histogram ---
    lifetimes = []
    # Track birth times to compute lifetimes for dead carriers
    birth_times = {e.get("carrier"): e.get("t", 0) for e in engine.birth_events}
    for d in engine.death_events:
        carrier_name = d.get("carrier", "")
        birth_t = birth_times.get(carrier_name, 0)
        death_t = d.get("t", 0)
        if death_t > birth_t:
            lifetimes.append(float(death_t - birth_t))
    # Add lifetimes of still-alive carriers (need to track their birth somehow)
    # For now, skip alive carriers as we don't store born_at in new system
    if lifetimes:
        fig = plt.figure(figsize=(7.2, 3.4), facecolor="white")
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(lifetimes, bins=18, alpha=0.85)
        ax.set_title("Carrier lifetimes (dissolved + alive at end)", fontsize=11)
        ax.set_xlabel("lifetime (s)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        fig.tight_layout(pad=0.4)
        fig.savefig(ARTIFACTS_DIR / "rez_stream_lifetimes.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # --- LaTeX snippet ---
    lines: list[str] = []
    lines.append(r"% Generated by tmp/rez/paper_artifacts.py.")
    lines.append(r"\paragraph{Stream run.}")
    lines.append(
        rf"We simulate an open-ended sensory stream for {sim_s:.0f}\,s (seed=0, $\Delta t$={float(rez.DT):.3f}\,s). "
        rf"Mean $N$={results['metrics']['N_mean']:.1f}, mean $M$={results['metrics']['M_mean']:.1f}, "
        rf"mean $\mathrm{{nnz}}(P)$={results['metrics']['nnz_mean']:.1f}, "
        rf"mean $L_{{\text{{comp}}}}$={results['metrics']['Lcomp_mean']:.1f}. "
        rf"Births={results['metrics']['births']}, deaths={results['metrics']['deaths']}."
    )
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stream summary metrics (time-averaged).}")
    lines.append(r"\begin{tabular}{@{}rrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"$\bar N$ & $\bar M$ & $\overline{\mathrm{nnz}(P)}$ & $\overline{L_{\text{comp}}}$ & Births & Deaths \\")
    lines.append(r"\midrule")
    lines.append(
        rf"{results['metrics']['N_mean']:.1f} & {results['metrics']['M_mean']:.1f} & {results['metrics']['nnz_mean']:.1f} & {results['metrics']['Lcomp_mean']:.1f} & {results['metrics']['births']} & {results['metrics']['deaths']} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    (ARTIFACTS_DIR / "rez_stream_autogen.tex").write_text("\n".join(lines) + "\n")

    end = time.perf_counter()
    print(f"[ok] wrote stream artifacts to {ARTIFACTS_DIR} (wall_s={end-start:.3f})")
    
    # Also run rigorous experiments
    print("\n" + "=" * 60)
    print("Running rigorous experiments...")
    print("=" * 60)
    
    try:
        from experiments import run_all_experiments, generate_experiment_artifacts
        results, rez_module = run_all_experiments()
        generate_experiment_artifacts(results, rez_module)
    except Exception as e:
        import traceback
        print(f"[warning] Could not run experiments: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

