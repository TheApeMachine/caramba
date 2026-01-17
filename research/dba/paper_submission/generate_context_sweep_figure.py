#!/usr/bin/env python3
"""Generate context sweep figure for the DBA paper."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUT_DIR = Path("/sessions/determined-epic-brahmagupta/mnt/dba/paper_submission")

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'baseline': '#1f77b4', 'dba': '#ff7f0e'}

# Data from benchmark run 2026-01-17
context_lens = [2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072]
baseline_toks = [11.6, 18.0, 12.4, 6.5, 0.64, 0.015, 0.017, 0.078]
dba_toks = [24.9, 24.2, 10.9, 8.4, 4.6, 2.4, 1.5, 1.05]

def generate_context_sweep_throughput():
    """Figure: Decode throughput vs context length (log-log)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(context_lens, baseline_toks, 'o-', color=COLORS['baseline'],
            linewidth=2, markersize=8, label='Baseline')
    ax.plot(context_lens, dba_toks, 's-', color=COLORS['dba'],
            linewidth=2, markersize=8, label='DBA (sem8/geo32)')

    ax.set_xlabel('Context Length (tokens)', fontsize=12)
    ax.set_ylabel('Decode Throughput (tok/s)', fontsize=12)
    ax.set_title('Single-Token Decode Throughput vs Context Length', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Add annotation for the crossover region
    ax.axvline(x=32768, color='gray', linestyle='--', alpha=0.5)
    ax.text(32768, 50, 'Baseline\ncollapses', fontsize=10, ha='center',
            color='gray', style='italic')

    # Shade the region where DBA wins big
    ax.axvspan(32768, 131072, alpha=0.1, color=COLORS['dba'])

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'context_sweep_throughput.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'context_sweep_throughput.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: context_sweep_throughput.png/pdf")


def generate_context_sweep_speedup():
    """Figure: DBA speedup ratio vs context length."""
    speedups = [dba / base for dba, base in zip(dba_toks, baseline_toks)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(context_lens)), speedups, color=COLORS['dba'],
                  edgecolor='black', linewidth=1)

    # Color the anomalous bar differently
    bars[2].set_color('#cccccc')  # 8k anomaly
    bars[2].set_edgecolor('black')

    ax.set_xticks(range(len(context_lens)))
    ax.set_xticklabels([f'{c//1024}k' for c in context_lens], fontsize=10)
    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('DBA Speedup (×)', fontsize=12)
    ax.set_title('DBA Speedup vs Baseline by Context Length', fontsize=14)

    # Add horizontal line at 1x
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        label = f'{speedup:.1f}×' if speedup < 10 else f'{speedup:.0f}×'
        y_pos = bar.get_height() + 2 if bar.get_height() < 100 else bar.get_height() * 1.05
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Annotate the anomaly
    ax.annotate('anomaly\n(noise)', xy=(2, speedups[2]), xytext=(2, 15),
                fontsize=9, ha='center', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.set_ylim(0, max(speedups) * 1.15)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'context_sweep_speedup.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'context_sweep_speedup.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: context_sweep_speedup.png/pdf")


def generate_combined_figure():
    """Figure: Combined throughput and speedup (2 panels)."""
    speedups = [dba / base for dba, base in zip(dba_toks, baseline_toks)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: throughput (log-log)
    ax1.plot(context_lens, baseline_toks, 'o-', color=COLORS['baseline'],
             linewidth=2, markersize=8, label='Baseline')
    ax1.plot(context_lens, dba_toks, 's-', color=COLORS['dba'],
             linewidth=2, markersize=8, label='DBA (sem8/geo32)')

    ax1.set_xlabel('Context Length (tokens)', fontsize=12)
    ax1.set_ylabel('Decode Throughput (tok/s)', fontsize=12)
    ax1.set_title('(a) Throughput vs Context Length', fontsize=13)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    # Add usability threshold line
    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.text(3000, 1.3, 'usability threshold (1 tok/s)', fontsize=9, color='red', alpha=0.8)

    # Right panel: speedup bars
    bar_colors = [COLORS['dba']] * len(speedups)
    bar_colors[2] = '#cccccc'  # 8k anomaly

    bars = ax2.bar(range(len(context_lens)), speedups, color=bar_colors,
                   edgecolor='black', linewidth=1)

    ax2.set_xticks(range(len(context_lens)))
    ax2.set_xticklabels([f'{c//1024}k' for c in context_lens], fontsize=10)
    ax2.set_xlabel('Context Length', fontsize=12)
    ax2.set_ylabel('DBA Speedup (×)', fontsize=12)
    ax2.set_title('(b) DBA Speedup vs Baseline', fontsize=13)

    ax2.axhline(y=1, color='black', linestyle='-', linewidth=1)

    # Value labels
    for bar, speedup in zip(bars, speedups):
        label = f'{speedup:.1f}×' if speedup < 10 else f'{speedup:.0f}×'
        y_pos = min(bar.get_height() + 3, 170)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_ylim(0, 180)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'context_sweep_combined.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'context_sweep_combined.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: context_sweep_combined.png/pdf")


if __name__ == "__main__":
    print(f"Generating context sweep figures to: {OUT_DIR}")
    generate_context_sweep_throughput()
    generate_context_sweep_speedup()
    generate_combined_figure()
    print("\nAll context sweep figures generated!")
