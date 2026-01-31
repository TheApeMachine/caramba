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
import sys
import wave
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Optional, Protocol

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tmp.rez.manifold import Manifold, ManifoldConfig

@dataclass
class SignalFrame:
    frequency: float
    amplitude: float
    phase: float
    duration: float


class SignalSource(Protocol):
    def get_signals(self, t: float, dt: float) -> list[SignalFrame]:
        ...


class AudioTokenizer:
    """Tokenize audio into SignalFrame lists."""

    def __init__(
        self,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        top_k: int = 12,
        min_energy: float = 0.02,
        amplitude_scale: float = 1.5,
    ):
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.top_k = top_k
        self.min_energy = min_energy
        self.amplitude_scale = amplitude_scale

    @staticmethod
    def _mixdown(audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            return audio.mean(dim=0)
        return audio

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    def tokenize(self, audio_stream: torch.Tensor, sample_rate: int) -> list[list[SignalFrame]]:
        audio = self._mixdown(audio_stream).to(dtype=torch.float32)
        if audio.numel() == 0:
            return []
        peak = float(audio.abs().max())
        if peak > 0:
            audio = audio / peak

        frame_len = max(16, int(round(self.frame_ms * sample_rate / 1000.0)))
        hop_len = max(8, int(round(self.hop_ms * sample_rate / 1000.0)))
        n_fft = self._next_pow2(frame_len)
        window = torch.hann_window(n_fft, device=audio.device)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_len,
            window=window,
            return_complex=True,
        )
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(audio.device)
        n_frames = spec.shape[1]
        frame_dt = frame_len / float(sample_rate)
        frames: list[list[SignalFrame]] = []

        for t in range(n_frames):
            frame_spec = spec[:, t]
            mags = frame_spec.abs()
            mags = mags.clone()
            mags[0] = 0.0
            frame_max = float(mags.max())
            if frame_max <= 0:
                frames.append([])
                continue
            threshold = frame_max * self.min_energy
            valid = mags >= threshold
            if not torch.any(valid):
                frames.append([])
                continue
            mags_valid = mags.clone()
            mags_valid[~valid] = 0.0
            k = min(self.top_k, mags_valid.numel())
            top_vals, top_idx = torch.topk(mags_valid, k=k)

            sigs: list[SignalFrame] = []
            for val, idx in zip(top_vals, top_idx):
                amp = float(val) / frame_max
                if amp <= 0:
                    continue
                sigs.append(
                    SignalFrame(
                        frequency=float(freqs[int(idx)]),
                        amplitude=amp * self.amplitude_scale,
                        phase=float(torch.angle(frame_spec[int(idx)])),
                        duration=frame_dt,
                    )
                )
            frames.append(sigs)
        return frames


def compute_tuning_strength(carrier_phases: torch.Tensor, osc_phases: torch.Tensor, gate_width: torch.Tensor) -> torch.Tensor:
    """Gaussian tuning strength T = exp(-(Δφ^2/σ))."""
    if carrier_phases.numel() == 0 or osc_phases.numel() == 0:
        return torch.empty(osc_phases.numel(), carrier_phases.numel())
    diff = (osc_phases[:, None] - carrier_phases[None, :]) % (2 * math.pi)
    diff = torch.where(diff > math.pi, diff - 2 * math.pi, diff)
    sigma = (gate_width / 2.0) ** 2
    return torch.exp(-(diff ** 2) / sigma[None, :])


class _OscState:
    def __init__(self, phases: torch.Tensor, amplitudes: torch.Tensor, omegas: torch.Tensor):
        self.phases = phases
        self.amplitudes = amplitudes
        self.omegas = omegas
        self.n = int(phases.numel())


class _CarrierState:
    def __init__(
        self,
        phases: torch.Tensor,
        energies: torch.Tensor,
        omegas: torch.Tensor,
        gate_widths: torch.Tensor,
        coherence: torch.Tensor,
    ):
        self.phases = phases
        self.energies = energies
        self.omegas = omegas
        self.gate_widths = gate_widths
        self.coherence_ema = coherence
        self.m = int(phases.numel())

    def gate(self) -> torch.Tensor:
        return (torch.cos(self.phases) >= 0).float()


class _PState:
    def __init__(self, presence: torch.Tensor):
        self.P = presence


class _ConfigState:
    def __init__(self, dt: float):
        self.dt = dt


class _EventsState:
    def __init__(self):
        self.births: list[dict] = []
        self.deaths: list[dict] = []
        self.mitoses: list[dict] = []


class ManifoldAdapter:
    """Expose manifold state with a minimal engine-like interface."""

    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        self.config = _ConfigState(dt=manifold.config.dt)
        self.events = _EventsState()
        self.refresh()

    def refresh(self) -> None:
        po = self.manifold.state
        osc = po.get("oscillators")
        carriers = po.get("carriers")
        bonds = po.get("bonds")
        self.t = self.manifold.t
        self.oscillators = _OscState(
            phases=osc.get("phase"),
            amplitudes=osc.get("amplitude"),
            omegas=2 * math.pi * osc.get("frequency"),
        )
        # Map Manifold carrier fields: heat -> energies, no coherence field (use zeros)
        n_carriers = carriers.shape[0]
        coherence = torch.zeros(n_carriers, dtype=torch.float32, device=carriers.get("phase").device)
        self.carriers = _CarrierState(
            phases=carriers.get("phase"),
            energies=carriers.get("heat"),  # Manifold uses "heat" instead of "energy"
            omegas=2 * math.pi * carriers.get("frequency"),
            gate_widths=carriers.get("gate_width"),
            coherence=coherence,  # Not tracked in Manifold, use zeros
        )
        self.P = _PState(bonds.get("presence"))

    def nnz_P(self) -> int:
        return self.manifold.nnz_P()

    def global_sync_R(self) -> float:
        return self.manifold.global_sync_R()

    def L_comp(self) -> float:
        return self.manifold.L_comp()
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
        signal_source: Optional[SignalSource] = None,
        manifold: Optional[Manifold] = None,
        history_len: int = 400,
    ):
        self.manifold = manifold or Manifold()
        self.signal_source = signal_source
        self.engine = ManifoldAdapter(self.manifold)
        
        # History for time series (last N steps)
        self.history_len = history_len
        self.t_history = deque(maxlen=history_len)
        self.N_history = deque(maxlen=history_len)
        self.M_history = deque(maxlen=history_len)
        self.nnz_history = deque(maxlen=history_len)
        self.R_history = deque(maxlen=history_len)
        self.L_history = deque(maxlen=history_len)
        
        self.fig: Optional[Figure] = None
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
        tuning = compute_tuning_strength(
            self.engine.carriers.phases,
            self.engine.oscillators.phases,
            self.engine.carriers.gate_widths,
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
        assert self.fig is not None, "Dashboard figure was not initialized."
        
        if self.engine.oscillators.n == 0 or self.engine.carriers.m == 0 or self.engine.P.P.numel() == 0:
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
        
        audio_line = ""
        if hasattr(self.signal_source, "frame_idx"):
            total_frames = len(getattr(self.signal_source, "frames", []))
            frame_idx = getattr(self.signal_source, "frame_idx", 0)
            sig_count = getattr(self.signal_source, "last_signals_count", 0)
            sig_energy = getattr(self.signal_source, "last_signal_energy", 0.0)
            loop_flag = getattr(self.signal_source, "loop", False)
            loop_text = "on" if loop_flag else "off"
            audio_line = (
                f"\nAUDIO\n  Frame: {frame_idx}/{total_frames}\n  "
                f"Signals: {sig_count}\n  Sig energy: {sig_energy:.2f}\n  Loop: {loop_text}"
            )

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
{audio_line}
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
            signals = self.signal_source.get_signals(self.engine.t, self.engine.config.dt) if self.signal_source else []
            payload = [
                {
                    "frequency": s.frequency,
                    "amplitude": s.amplitude,
                    "phase": s.phase,
                    "duration": s.duration,
                }
                for s in signals
            ]
            self.manifold.step(payload)
            self.engine.refresh()
        
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
        assert self.fig is not None, "Dashboard figure was not initialized."
        self.ani = FuncAnimation(
            self.fig, self.update, 
            interval=50,  # 20 FPS
            blit=False,
            cache_frame_data=False
        )
        plt.show()


class AudioFrameSource:
    """Signal source that yields per-frame signals from audio."""

    def __init__(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        *,
        engine_dt: float,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        top_k: int = 12,
        min_energy: float = 0.02,
        amplitude_scale: float = 1.5,
        loop: bool = True,
    ):
        self.tokenizer = AudioTokenizer(
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            top_k=top_k,
            min_energy=min_energy,
            amplitude_scale=amplitude_scale,
        )
        self.frames = self.tokenizer.tokenize(audio, sample_rate)
        self.frame_dt = hop_ms / 1000.0
        self.steps_per_frame = max(1, int(round(self.frame_dt / engine_dt)))
        self.frame_idx = 0
        self.substep = 0
        self.last_signals_count = 0
        self.last_signal_energy = 0.0
        self.loop = loop

    def get_signals(self, t: float, dt: float) -> list[SignalFrame]:
        if self.frame_idx >= len(self.frames):
            if self.loop and self.frames:
                self.frame_idx = 0
            else:
                return []
        signals = self.frames[self.frame_idx] if self.substep == 0 else []
        if self.substep == 0:
            self.last_signals_count = len(signals)
            self.last_signal_energy = float(sum(sig.amplitude for sig in signals)) if signals else 0.0
        self.substep += 1
        if self.substep >= self.steps_per_frame:
            self.substep = 0
            self.frame_idx += 1
        return signals


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
    
    manifold = Manifold()
    dashboard = Dashboard(manifold=manifold)
    dashboard.run()


def main_two_speakers():
    """Launch the dashboard on two-speaker audio."""
    print("=" * 60)
    print("Resonant Compression Systems — Audio Dashboard (Two Speakers)")
    print("=" * 60)
    print()
    print("Key:")
    print("  Phase Circle: Shows WHERE oscillators (dots) and carriers (squares) are")
    print("  Alignment: Shows HOW WELL each oscillator matches each carrier (antenna principle)")
    print("  Bonds P: Shows WHO is connected to WHOM")
    print("  Orange = gate OPEN (carrier is listening)")
    print("  Gray = gate CLOSED (carrier is not listening)")
    print()

    manifold = Manifold()
    def _load_wav_fallback(path: str) -> tuple[torch.Tensor, int]:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            dtype = np.uint8
            audio = np.frombuffer(raw, dtype=dtype).astype(np.int16) - 128
            max_val = 128.0
        elif sampwidth == 2:
            dtype = np.int16
            audio = np.frombuffer(raw, dtype=dtype)
            max_val = 32768.0
        elif sampwidth == 3:
            data = np.frombuffer(raw, dtype=np.uint8)
            audio = data.reshape(-1, 3)
            audio = (audio[:, 0].astype(np.int32) |
                     (audio[:, 1].astype(np.int32) << 8) |
                     (audio[:, 2].astype(np.int32) << 16))
            audio = (audio.astype(np.int32) << 8) >> 8
            max_val = float(1 << 23)
        elif sampwidth == 4:
            dtype = np.int32
            audio = np.frombuffer(raw, dtype=dtype)
            max_val = float(1 << 31)
        else:
            raise RuntimeError(f"Unsupported sample width: {sampwidth}")

        audio = audio.astype(np.float32) / max_val
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).T
        else:
            audio = audio.reshape(1, -1)
        return torch.from_numpy(audio), int(sample_rate)

    try:
        audio, sr = torchaudio.load("tmp/rez/two_speakers.wav")
    except Exception:
        audio, sr = _load_wav_fallback("tmp/rez/two_speakers.wav")

    source = AudioFrameSource(audio, sr, engine_dt=manifold.config.dt)
    dashboard = Dashboard(source, manifold=manifold)
    dashboard.run()


if __name__ == "__main__":
    main_two_speakers()
