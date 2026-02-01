#!/usr/bin/env python3
"""
Semantic Manifold Dashboard
===========================

Real-time visualization for the thermodynamic grammar system.
Focuses on causal chains: excitation -> heat -> entropy -> grammar flow.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from semantic import SemanticManifold
from physics import PhysicsConfig as SemanticPhysicsConfig, DTYPE_REAL


@dataclass
class DashboardConfig:
    vocab_size: int = 64
    embed_dim: int = 64
    dt: float = 0.1
    history_len: int = 300
    steps_per_frame: int = 2
    num_modes: int = 4


class SemanticDashboard:
    def __init__(self, config: DashboardConfig):
        self.cfg = config
        self.device = torch.device("cpu")
        sem_cfg = SemanticPhysicsConfig(dt=config.dt)
        self.brain = SemanticManifold(
            sem_cfg,
            self.device,
            config.embed_dim,
            config.vocab_size,
            num_modes=config.num_modes,
        )
        # Seed with an initial context
        embeddings = self.brain.attractors.get("position")
        self.brain.ingest_context(embeddings[: min(8, config.vocab_size)])

        # History buffers
        self.t = []
        self.entropy = []
        self.confidence = []
        self.heat_mean = []
        self.heat_var = []
        self.income = []
        self.grammar_flow = []
        self.adj_nnz = []
        self.mode_entropy = []

        self.fig: Optional[Figure] = None
        self.ani = None
        self._build()

    def _build(self) -> None:
        self.fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.06, wspace=0.2, hspace=0.25)

        gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[1.0, 1.0])
        self.ax_series = self.fig.add_subplot(gs[0, 0])
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_bonds = self.fig.add_subplot(gs[1, 0])
        self.ax_stats = self.fig.add_subplot(gs[1, 1])

        for ax in [self.ax_series, self.ax_heat, self.ax_bonds, self.ax_stats]:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

    def _append_history(self, debug: dict) -> None:
        step = len(self.t)
        self.t.append(step * self.cfg.dt)
        self.entropy.append(float(debug.get("entropy_after", 0.0)))
        self.confidence.append(self.brain.thinking_confidence())
        self.heat_mean.append(float(debug.get("heat_mean", 0.0)))
        self.heat_var.append(float(debug.get("heat_scale", 0.0)))
        self.income.append(float(debug.get("income_abs_mean", 0.0)))
        self.grammar_flow.append(float(debug.get("grammatical_flow_mean", 0.0)))
        self.adj_nnz.append(float(debug.get("adjacency_nonzero", 0.0)))
        self.mode_entropy.append(float(self.brain.last_mode_entropy or 0.0))

        max_len = self.cfg.history_len
        if len(self.t) > max_len:
            for arr in [
                self.t,
                self.entropy,
                self.confidence,
                self.heat_mean,
                self.heat_var,
                self.income,
                self.grammar_flow,
                self.adj_nnz,
                self.mode_entropy,
            ]:
                del arr[: len(arr) - max_len]

    def _draw_series(self) -> None:
        ax = self.ax_series
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.set_title("Entropy / Confidence / Mode Entropy", color="white", fontsize=9, fontweight="bold")
        if len(self.t) < 2:
            return
        ax.plot(self.t, self.entropy, color="#ff6b6b", linewidth=1, label="Entropy")
        ax.plot(self.t, self.confidence, color="#4ecdc4", linewidth=1, label="Confidence")
        ax.plot(self.t, self.mode_entropy, color="#ffe66d", linewidth=1, label="Mode entropy")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.4)
        ax.grid(True, alpha=0.15)

    def _draw_heat(self) -> None:
        ax = self.ax_heat
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.set_title("Heat / Income / Grammar Flow", color="white", fontsize=9, fontweight="bold")
        if len(self.t) < 2:
            return
        ax.plot(self.t, self.heat_mean, color="#ffbf69", linewidth=1, label="Heat mean")
        ax.plot(self.t, self.heat_var, color="#fcca46", linewidth=1, label="Heat scale")
        ax.plot(self.t, self.income, color="#a8e6cf", linewidth=1, label="Income |mean|")
        ax.plot(self.t, self.grammar_flow, color="#c3aed6", linewidth=1, label="Grammar flow")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.4)
        ax.grid(True, alpha=0.15)

    def _draw_bonds(self) -> None:
        ax = self.ax_bonds
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.set_title("Transition Matrix (active view)", color="white", fontsize=9, fontweight="bold")
        tm = self.brain.transition_matrix.detach().cpu().numpy()
        if tm.size == 0:
            return
        im = ax.imshow(tm, cmap="viridis", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        if self.fig is not None:
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _draw_stats(self, debug: dict) -> None:
        ax = self.ax_stats
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        stats = [
            f"Adj nnz: {debug.get('adjacency_nonzero', 0.0):.4f}",
            f"Adj mean: {debug.get('adjacency_mean', 0.0):.6f}",
            f"Exc scale: {debug.get('exc_scale', 0.0):.4f}",
            f"Exc mean: {debug.get('exc_active_mean', 0.0):.4f}",
            f"Heat mean: {debug.get('heat_mean', 0.0):.6f}",
            f"Heat util: {debug.get('heat_utility', 0.0):.6f}",
            f"Income mean: {debug.get('income_mean', 0.0):.6f}",
            f"Income |mean|: {debug.get('income_abs_mean', 0.0):.6f}",
            f"Flow scale: {debug.get('flow_scale', 0.0):.6f}",
            f"Grammar flow: {debug.get('grammatical_flow_mean', 0.0):.6f}",
        ]
        ax.text(
            0.03,
            0.97,
            "\n".join(stats),
            transform=ax.transAxes,
            fontsize=7,
            color="white",
            family="monospace",
            verticalalignment="top",
        )

    def update(self, _frame):
        for _ in range(self.cfg.steps_per_frame):
            self.brain.step_grammar()
            debug = getattr(self.brain, "last_debug", {})
            self._append_history(debug)

        debug = getattr(self.brain, "last_debug", {})
        self._draw_series()
        self._draw_heat()
        self._draw_bonds()
        self._draw_stats(debug)
        return []

    def run(self) -> None:
        assert self.fig is not None
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False, cache_frame_data=False)
        plt.show()


def main():
    cfg = DashboardConfig()
    dashboard = SemanticDashboard(cfg)
    dashboard.run()


if __name__ == "__main__":
    main()
