"""
Real-Time Visualization Dashboard for Resonant Compression Systems
===================================================================

Designed for understanding, not just display.
Compact layout with minimal wasted space.

Key visualizations:
1. Oscillator phases on unit circle (where are they?)
2. Carrier phases on unit circle (where are the antennas pointed?)
3. Presence matrix heatmap (who is connected to whom?)
4. Alignment matrix (how well does each oscillator match each carrier?)
5. Time series of key metrics (what's happening over time?)
6. Gate status (which carriers are listening right now?)
7. Energy levels (how strong is each carrier?)
"""

import math
from collections import deque
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from main import (
    ResonantEngine,
    StochasticStream,
    PhysicsConfig,
    Signal,
    compute_alignment,
    compute_tuning_strength_per_carrier,
    DEVICE,
    DTYPE_REAL,
)


class Dashboard:
    """
    Real-time visualization dashboard.
    
    Shows the system state in an intuitive way:
    - Phase circles show WHERE oscillators and carriers are
    - Presence matrix shows WHO is connected
    - Alignment shows HOW WELL they match (the antenna principle!)
    - Time series show WHAT'S CHANGING
    """
    
    def __init__(
        self,
        engine: ResonantEngine,
        signal_source: Optional[StochasticStream] = None,
        history_len: int = 400,
    ):
        self.engine = engine
        self.signal_source = signal_source or StochasticStream(seed=0)
        
        # History for time series (last N steps)
        self.history_len = history_len
        self.t_history = deque(maxlen=history_len)
        self.N_history = deque(maxlen=history_len)
        self.M_history = deque(maxlen=history_len)
        self.nnz_history = deque(maxlen=history_len)
        self.R_history = deque(maxlen=history_len)
        self.L_history = deque(maxlen=history_len)
        
        self.fig = None
        self.ani = None
        self._build()
    
    def _build(self):
        """Build the dashboard layout."""
        # Compact figure - no wasted space
        self.fig = plt.figure(figsize=(16, 9), facecolor='#1a1a2e')
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, 
                                  wspace=0.15, hspace=0.25)
        
        # Grid layout:
        # Row 0: [Phase Circles (osc+carrier)] [Alignment Matrix] [Presence Matrix]
        # Row 1: [Time Series: N, M, nnz]      [Time Series: R, L] [Gate/Energy bars]
        # Row 2: [Signal traces]               [Carrier traces]    [Stats text]
        
        gs = gridspec.GridSpec(3, 3, figure=self.fig, height_ratios=[1.2, 0.8, 1.0])
        
        # Row 0
        self.ax_phases = self.fig.add_subplot(gs[0, 0])
        self.ax_alignment = self.fig.add_subplot(gs[0, 1])
        self.ax_presence = self.fig.add_subplot(gs[0, 2])
        
        # Row 1
        self.ax_counts = self.fig.add_subplot(gs[1, 0])
        self.ax_metrics = self.fig.add_subplot(gs[1, 1])
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        
        # Row 2
        self.ax_signals = self.fig.add_subplot(gs[2, 0])
        self.ax_carriers = self.fig.add_subplot(gs[2, 1])
        self.ax_stats = self.fig.add_subplot(gs[2, 2])
        
        # Style all axes
        for ax in [self.ax_phases, self.ax_alignment, self.ax_presence,
                   self.ax_counts, self.ax_metrics, self.ax_energy,
                   self.ax_signals, self.ax_carriers, self.ax_stats]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#30363d')
    
    def _draw_phase_circle(self, ax):
        """
        Draw oscillator and carrier phases on a unit circle.
        
        This shows WHERE each oscillator and carrier is in its cycle.
        Think of it like a clock - all the hands pointing at different times.
        """
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Phase Circle (where in cycle)', color='white', fontsize=9, fontweight='bold')
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'gray', alpha=0.3, linewidth=1)
        ax.axhline(0, color='gray', alpha=0.2, linewidth=0.5)
        ax.axvline(0, color='gray', alpha=0.2, linewidth=0.5)
        
        # Draw oscillators as small dots on circle
        if self.engine.oscillators.n > 0:
            phases = self.engine.oscillators.phases.cpu().numpy()
            amps = self.engine.oscillators.amplitudes.cpu().numpy()
            
            # Normalize amplitude for size (min 20, max 100)
            sizes = 20 + 80 * np.clip(amps, 0, 1)
            
            x = np.cos(phases)
            y = np.sin(phases)
            ax.scatter(x, y, s=sizes, c='#4ecdc4', alpha=0.7, label=f'Osc ({self.engine.oscillators.n})')
        
        # Draw carriers as wedges (showing gate!)
        if self.engine.carriers.m > 0:
            phases = self.engine.carriers.phases.cpu().numpy()
            gates = self.engine.carriers.gate().cpu().numpy()
            energies = self.engine.carriers.energies.cpu().numpy()
            
            for i, (phase, gate, energy) in enumerate(zip(phases, gates, energies)):
                # Carrier position
                r = 1.2  # Slightly outside the circle
                x = r * np.cos(phase)
                y = r * np.sin(phase)
                
                # Color based on gate state
                color = '#ffbf69' if gate > 0.5 else '#666666'
                
                # Size based on energy
                size = 50 + 100 * min(energy, 1)
                
                ax.scatter([x], [y], s=size, c=color, marker='s', alpha=0.8)
                
                # Draw gate arc (shows when it's open)
                # Gate is open when cos(phase) >= 0, i.e., phase in [-π/2, π/2]
                if gate > 0.5:
                    # Draw arc showing gate window
                    gate_start = np.degrees(phase - np.pi/2)
                    gate_end = np.degrees(phase + np.pi/2)
                    wedge = Wedge((0, 0), 1.0, gate_start, gate_end, 
                                  width=0.1, facecolor=color, alpha=0.15)
                    ax.add_patch(wedge)
        
        ax.legend(loc='upper right', fontsize=7, framealpha=0.5)
        ax.axis('off')
    
    def _draw_alignment_matrix(self, ax):
        """
        Draw TUNING STRENGTH between oscillators and carriers.
        
        This is the RADIO DIAL visualization (Gaussian tuning, not cosine):
        - Bright: strong tuning (clear signal, strong coupling)
        - Dark: weak tuning (no signal, noise only)
        
        Now uses per-carrier gate widths for emergent specialization.
        """
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Tuning T', color='white', fontsize=9, fontweight='bold')
        
        if self.engine.oscillators.n == 0 or self.engine.carriers.m == 0:
            ax.text(0.5, 0.5, 'No tuning', 
                    ha='center', va='center', color='gray', fontsize=9,
                    transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Compute tuning strength with per-carrier gate widths
        tuning = compute_tuning_strength_per_carrier(
            self.engine.carriers.phases,
            self.engine.oscillators.phases,
            self.engine.carriers.gate_widths
        ).cpu().numpy()
        
        # Show as heatmap - no colorbar (bonds panel has it)
        im = ax.imshow(tuning, cmap='plasma', vmin=0, vmax=1, aspect='auto')
        
        # Minimal labels - no axis labels, sparse ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Just show carrier count at bottom
        ax.set_xlabel(f'{self.engine.carriers.m} carriers', color='gray', fontsize=7)
    
    def _draw_presence_matrix(self, ax):
        """
        Draw presence matrix P (bond strengths).
        
        Shows WHO is connected to WHOM and HOW STRONGLY.
        Brighter = stronger bond.
        This panel has the colorbar since it's on the right.
        """
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Bonds P', color='white', fontsize=9, fontweight='bold')
        
        if self.engine.P.P.numel() == 0:
            ax.text(0.5, 0.5, 'No bonds', 
                    ha='center', va='center', color='gray', fontsize=9,
                    transform=ax.transAxes)
            ax.axis('off')
            return
        
        P = self.engine.P.P.cpu().numpy()
        
        im = ax.imshow(P, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        
        # Only this rightmost panel gets axis labels
        ax.set_xlabel('Carrier', color='white', fontsize=7)
        ax.set_ylabel('Osc', color='white', fontsize=7)
        
        # Sparse ticks - only show a few
        n_carriers = self.engine.carriers.m
        if n_carriers <= 6:
            ax.set_xticks(range(n_carriers))
            ax.set_xticklabels(range(n_carriers), fontsize=6)
        else:
            # Show only first, middle, last
            ticks = [0, n_carriers // 2, n_carriers - 1]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, fontsize=6)
        
        n_osc = self.engine.oscillators.n
        if n_osc <= 6:
            ax.set_yticks(range(n_osc))
            ax.set_yticklabels(range(n_osc), fontsize=6)
        else:
            ticks = [0, n_osc // 2, n_osc - 1]
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks, fontsize=6)
        
        # Colorbar only on this panel
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white', labelsize=6)
        cbar.set_label('P', color='white', fontsize=7)
    
    def _draw_counts(self, ax):
        """Draw time series of N (oscillators), M (carriers), nnz (bonds)."""
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Population over time', color='white', fontsize=9, fontweight='bold')
        
        if len(self.t_history) < 2:
            ax.text(0.5, 0.5, 'Collecting data...', 
                    ha='center', va='center', color='gray', fontsize=9,
                    transform=ax.transAxes)
            return
        
        t = list(self.t_history)
        ax.plot(t, list(self.N_history), color='#4ecdc4', linewidth=1, label='Oscillators')
        ax.plot(t, list(self.M_history), color='#ffbf69', linewidth=1, label='Carriers')
        ax.plot(t, list(self.nnz_history), color='#a8e6cf', linewidth=1, label='Bonds', linestyle='--')
        
        ax.legend(loc='upper left', fontsize=6, framealpha=0.5)
        ax.set_xlabel('Time (s)', color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        ax.grid(True, alpha=0.1)
    
    def _draw_metrics(self, ax):
        """Draw time series of R (sync) and L_comp (compression)."""
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Sync R & Compression L', color='white', fontsize=9, fontweight='bold')
        
        if len(self.t_history) < 2:
            return
        
        t = list(self.t_history)
        
        # R on left axis (0 to 1)
        ax.plot(t, list(self.R_history), color='#ff6b6b', linewidth=1, label='R (sync)')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('R', color='#ff6b6b', fontsize=7)
        ax.tick_params(axis='y', colors='#ff6b6b', labelsize=6)
        
        # L on right axis
        ax2 = ax.twinx()
        ax2.plot(t, list(self.L_history), color='#c3aed6', linewidth=1, label='L_comp')
        ax2.set_ylabel('L_comp', color='#c3aed6', fontsize=7)
        ax2.tick_params(axis='y', colors='#c3aed6', labelsize=6)
        ax2.spines['right'].set_color('#c3aed6')
        
        ax.set_xlabel('Time (s)', color='white', fontsize=7)
        ax.tick_params(axis='x', colors='white', labelsize=6)
        ax.grid(True, alpha=0.1)
    
    def _draw_energy_bars(self, ax):
        """
        Draw carrier energies, coherence, and gate status as grouped bars.
        
        Shows:
        - Left bar (solid) = carrier energy
        - Right bar (hatched) = coherence EMA
        - Color = gate open (orange) or closed (gray)
        """
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Energy & Coh', color='white', fontsize=9, fontweight='bold')
        
        if self.engine.carriers.m == 0:
            ax.text(0.5, 0.5, 'No carriers', 
                    ha='center', va='center', color='gray', fontsize=9,
                    transform=ax.transAxes)
            return
        
        energies = self.engine.carriers.energies.cpu().numpy()
        coherences = self.engine.carriers.coherence_ema.cpu().numpy()
        gates = self.engine.carriers.gate().cpu().numpy()
        
        n = len(energies)
        x = np.arange(n)
        width = 0.35
        
        # Energy bars (left)
        colors_energy = ['#ffbf69' if g > 0.5 else '#666666' for g in gates]
        ax.bar(x - width/2, energies, width, color=colors_energy, edgecolor='white', 
               linewidth=0.5, label='E')
        
        # Coherence bars (right) - use lighter version of same color
        colors_coh = ['#ffe4b5' if g > 0.5 else '#999999' for g in gates]
        ax.bar(x + width/2, coherences, width, color=colors_coh, edgecolor='white',
               linewidth=0.5, label='C', hatch='//')
        
        # Sparse x-axis labels
        if n <= 8:
            ax.set_xticks(x)
            ax.set_xticklabels(range(n), fontsize=6)
        else:
            # Only show a few tick marks
            ticks = [0, n // 2, n - 1]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, fontsize=6)
        
        ax.tick_params(colors='white', labelsize=6)
        ax.set_ylim(0, max(1.2, energies.max() * 1.2, coherences.max() * 1.2) if n > 0 else 1.2)
        ax.legend(loc='upper right', fontsize=6, framealpha=0.5)
    
    def _draw_stats(self, ax):
        """Draw text statistics including specialization metrics."""
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.axis('off')
        
        # Compute stats
        N = self.engine.oscillators.n
        M = self.engine.carriers.m
        nnz = self.engine.nnz_P()
        R = self.engine.global_sync_R()
        L = self.engine.L_comp()
        
        naive = N * M if M > 0 else 0
        ratio = L / naive if naive > 0 else 0
        
        births = len(self.engine.events.births)
        deaths = len(self.engine.events.deaths)
        mitoses = len(self.engine.events.mitoses)
        
        # Specialization metrics
        if M > 0:
            gate_widths = self.engine.carriers.gate_widths.cpu().numpy()
            coherences = self.engine.carriers.coherence_ema.cpu().numpy()
            mean_gw = np.mean(gate_widths) / np.pi
            min_gw = np.min(gate_widths) / np.pi
            mean_coh = np.mean(coherences)
            spec_line = f"""
SPECIALIZATION
  Mean gate: {mean_gw:.2f}π
  Min gate:  {min_gw:.2f}π
  Mean coh:  {mean_coh:.2f}"""
        else:
            spec_line = ""
        
        stats = f"""TIME: {self.engine.t:.2f}s

POPULATION
  Oscillators: {N}
  Carriers: {M}
  Bonds: {nnz}

COMPRESSION
  L_comp: {L}
  Naive: {naive}
  Ratio: {ratio:.2f}

SYNC: R = {R:.3f}
{spec_line}

EVENTS
  Births: {births}
  Deaths: {deaths}
  Mitoses: {mitoses}
"""
        
        ax.text(0.05, 0.95, stats, transform=ax.transAxes,
                fontsize=7, color='white', family='monospace',
                verticalalignment='top')
    
    def _draw_signals(self, ax):
        """Draw oscillator waveforms (simplified view)."""
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Oscillator Signals (last few)', color='white', fontsize=9, fontweight='bold')
        
        if self.engine.oscillators.n == 0:
            ax.text(0.5, 0.5, 'No oscillators', 
                    ha='center', va='center', color='gray', fontsize=9,
                    transform=ax.transAxes)
            return
        
        # Show up to 8 oscillators
        n_show = min(8, self.engine.oscillators.n)
        phases = self.engine.oscillators.phases[:n_show].cpu().numpy()
        amps = self.engine.oscillators.amplitudes[:n_show].cpu().numpy()
        freqs = self.engine.oscillators.omegas[:n_show].cpu().numpy() / (2 * np.pi)
        
        # Generate waveforms for display
        t = np.linspace(0, 0.5, 100)
        for i in range(n_show):
            wave = amps[i] * np.cos(2 * np.pi * freqs[i] * t + phases[i])
            offset = i * 2.5
            ax.plot(t, wave + offset, color='#4ecdc4', linewidth=0.8, alpha=0.8)
            ax.text(-0.02, offset, f'{freqs[i]:.1f}Hz', color='white', fontsize=6, ha='right')
        
        ax.set_xlim(-0.08, 0.5)
        ax.set_xlabel('Time (s)', color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        ax.set_yticks([])
    
    def _draw_carriers(self, ax):
        """Draw carrier waveforms with gate."""
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title('Carrier Gates (capture windows)', color='white', fontsize=9, fontweight='bold')
        
        if self.engine.carriers.m == 0:
            ax.text(0.5, 0.5, 'No carriers', 
                    ha='center', va='center', color='gray', fontsize=9,
                    transform=ax.transAxes)
            return
        
        # Show up to 6 carriers
        n_show = min(6, self.engine.carriers.m)
        phases = self.engine.carriers.phases[:n_show].cpu().numpy()
        omegas = self.engine.carriers.omegas[:n_show].cpu().numpy()
        energies = self.engine.carriers.energies[:n_show].cpu().numpy()
        
        t = np.linspace(0, 1, 200)
        for i in range(n_show):
            # Carrier oscillation
            carrier_phase = phases[i] + omegas[i] * t
            wave = energies[i] * np.cos(carrier_phase)
            
            # Gate (open when cos >= 0)
            gate = (np.cos(carrier_phase) >= 0).astype(float)
            
            offset = i * 2
            ax.fill_between(t, offset - 0.4, offset + 0.4, where=gate > 0.5,
                           color='#ffbf69', alpha=0.2)
            ax.plot(t, wave * 0.8 + offset, color='#ffbf69', linewidth=0.8)
            ax.text(-0.02, offset, f'C{i}', color='white', fontsize=6, ha='right')
        
        ax.set_xlim(-0.06, 1)
        ax.set_xlabel('Time (s)', color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        ax.set_yticks([])
    
    def update(self, frame):
        """Update the dashboard for one frame."""
        # Run several simulation steps per frame for smoother animation
        for _ in range(4):
            signals = self.signal_source.get_signals(self.engine.t, self.engine.config.dt)
            self.engine.step(signals)
        
        # Record history
        self.t_history.append(self.engine.t)
        self.N_history.append(self.engine.oscillators.n)
        self.M_history.append(self.engine.carriers.m)
        self.nnz_history.append(self.engine.nnz_P())
        self.R_history.append(self.engine.global_sync_R())
        self.L_history.append(self.engine.L_comp())
        
        # Redraw all panels
        self._draw_phase_circle(self.ax_phases)
        self._draw_alignment_matrix(self.ax_alignment)
        self._draw_presence_matrix(self.ax_presence)
        self._draw_counts(self.ax_counts)
        self._draw_metrics(self.ax_metrics)
        self._draw_energy_bars(self.ax_energy)
        self._draw_signals(self.ax_signals)
        self._draw_carriers(self.ax_carriers)
        self._draw_stats(self.ax_stats)
        
        return []
    
    def run(self):
        """Start the dashboard."""
        self.ani = FuncAnimation(
            self.fig, self.update, 
            interval=50,  # 20 FPS
            blit=False,
            cache_frame_data=False
        )
        plt.show()


def main():
    """Launch the dashboard."""
    print("=" * 60)
    print("Resonant Compression Systems — Real-Time Dashboard")
    print("=" * 60)
    print()
    print("Key:")
    print("  Phase Circle: Shows WHERE oscillators (dots) and carriers (squares) are")
    print("  Alignment: Shows HOW WELL each oscillator matches each carrier (antenna principle)")
    print("  Bonds P: Shows WHO is connected to WHOM")
    print("  Orange = gate OPEN (carrier is listening)")
    print("  Gray = gate CLOSED (carrier is not listening)")
    print()
    
    engine = ResonantEngine(seed=0)
    stream = StochasticStream(seed=0)
    
    dashboard = Dashboard(engine, stream)
    dashboard.run()


if __name__ == "__main__":
    main()
