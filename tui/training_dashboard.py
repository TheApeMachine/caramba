"""Training Dashboard TUI with real-time metric plotting.

This module provides training dashboard components that can be used standalone
or as part of the unified TUI (see unified.py).

Components:
- TrainingMetrics: Data container for training metrics history
- MetricPlot: Individual plot widget for a metric
- MetricsPanel: Current metrics display panel
- TimingBar: Visual timing breakdown bar
- TrainingDashboard: Standalone dashboard app (prefer unified.py)

For the unified TUI with all views, use:
    caramba tui

Uses textual-plotext for terminal-based plotting.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

try:
    from textual_plotext import PlotextPlot
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    PlotextPlot = None  # type: ignore[misc,assignment]


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""

    max_points: int = 200

    # Time series data
    steps: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    loss: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    ppl: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    lr: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    grad_norm: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    tok_s: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    mem_params_mb: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    mem_grads_mb: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    mem_optim_mb: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    # Current values
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    current_ppl: float = 0.0
    current_lr: float = 0.0
    current_tok_s: float = 0.0

    # Timing
    ms_fwd: float = 0.0
    ms_bwd: float = 0.0
    ms_opt: float = 0.0
    ms_step: float = 0.0

    def update(self, step: int, metrics: dict[str, float]) -> None:
        """Update metrics from a training step."""
        self.current_step = step
        self.steps.append(step)

        if "loss" in metrics:
            self.loss.append(metrics["loss"])
            self.current_loss = metrics["loss"]
        if "ppl" in metrics:
            self.ppl.append(min(metrics["ppl"], 1e6))  # Cap for display
            self.current_ppl = metrics["ppl"]
        if "lr" in metrics:
            self.lr.append(metrics["lr"])
            self.current_lr = metrics["lr"]
        if "grad_norm" in metrics:
            self.grad_norm.append(min(metrics["grad_norm"], 100))  # Cap outliers
        if "tok_s" in metrics:
            self.tok_s.append(metrics["tok_s"])
            self.current_tok_s = metrics["tok_s"]
        if "mem_params_mb" in metrics:
            self.mem_params_mb.append(metrics["mem_params_mb"])
        if "mem_grads_mb" in metrics:
            self.mem_grads_mb.append(metrics["mem_grads_mb"])
        if "mem_optim_mb" in metrics:
            self.mem_optim_mb.append(metrics["mem_optim_mb"])

        # Timing
        self.ms_fwd = metrics.get("ms_fwd", 0.0)
        self.ms_bwd = metrics.get("ms_bwd", 0.0)
        self.ms_opt = metrics.get("ms_opt", 0.0)
        self.ms_step = metrics.get("ms_step", 0.0)


class MetricPlot(Static):
    """A single metric plot widget."""

    DEFAULT_CSS = """
    MetricPlot {
        height: 100%;
        width: 100%;
        border: round #45475A;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        title: str,
        metric_key: str,
        color: str = "cyan",
        y_label: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.metric_key = metric_key
        self.color = color
        self.y_label = y_label or metric_key
        self._plot: Any = None

    def compose(self) -> ComposeResult:
        if PLOTEXT_AVAILABLE:
            yield PlotextPlot()  # type: ignore[misc]
        else:
            yield Label(f"[dim]Plot: {self.title}[/]\n[yellow]textual-plotext not installed[/]")

    def on_mount(self) -> None:
        if PLOTEXT_AVAILABLE:
            try:
                self._plot = self.query_one("PlotextPlot")  # type: ignore[assignment]
            except Exception:
                pass

    def update_plot(self, steps: list[int], values: list[float]) -> None:
        """Update the plot with new data."""
        if not PLOTEXT_AVAILABLE or self._plot is None:
            return
        if not steps or not values:
            return

        plt = getattr(self._plot, "plt", None)
        if plt is None:
            return
        plt.clear_figure()
        plt.theme("textual-design-dark")
        plt.plot(steps, values, marker="braille", color=self.color)
        plt.title(self.title)
        plt.xlabel("Step")
        plt.ylabel(self.y_label)
        if hasattr(self._plot, "refresh"):
            self._plot.refresh()  # type: ignore[union-attr]


class MetricsPanel(Static):
    """Panel showing current metric values."""

    DEFAULT_CSS = """
    MetricsPanel {
        height: auto;
        padding: 1;
        background: #2D2D3D;
        border: round #45475A;
    }

    MetricsPanel .metric-row {
        height: 1;
    }

    MetricsPanel .metric-label {
        width: 20;
        color: #6C7086;
    }

    MetricsPanel .metric-value {
        color: #7aa2f7;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._labels: dict[str, Label] = {}

    def compose(self) -> ComposeResult:
        metrics = [
            ("step", "Step"),
            ("loss", "Loss"),
            ("ppl", "Perplexity"),
            ("lr", "Learning Rate"),
            ("tok_s", "Tokens/sec"),
            ("ms_step", "Step Time (ms)"),
            ("grad_norm", "Grad Norm"),
        ]
        for key, label in metrics:
            with Horizontal(classes="metric-row"):
                yield Label(f"{label}:", classes="metric-label")
                lbl = Label("--", classes="metric-value", id=f"metric-{key}")
                self._labels[key] = lbl
                yield lbl

    def update_metrics(self, m: TrainingMetrics) -> None:
        """Update displayed metrics."""
        updates = {
            "step": f"{m.current_step:,} / {m.total_steps:,}" if m.total_steps else f"{m.current_step:,}",
            "loss": f"{m.current_loss:.6f}",
            "ppl": f"{m.current_ppl:.2f}" if m.current_ppl < 1e6 else "inf",
            "lr": f"{m.current_lr:.2e}",
            "tok_s": f"{m.current_tok_s:,.0f}",
            "ms_step": f"{m.ms_step:.1f}",
            "grad_norm": f"{m.grad_norm[-1]:.4f}" if m.grad_norm else "--",
        }
        for key, value in updates.items():
            if key in self._labels:
                self._labels[key].update(value)


class TimingBar(Static):
    """Visual timing breakdown bar."""

    DEFAULT_CSS = """
    TimingBar {
        height: 3;
        padding: 0 1;
        background: #2D2D3D;
        border: round #45475A;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def update_timing(self, ms_fwd: float, ms_bwd: float, ms_opt: float) -> None:
        """Update timing breakdown."""
        total = ms_fwd + ms_bwd + ms_opt
        if total <= 0:
            self.update("[dim]No timing data[/]")
            return

        fwd_pct = ms_fwd / total * 100
        bwd_pct = ms_bwd / total * 100
        opt_pct = ms_opt / total * 100

        # Simple text-based bar
        bar_width = 40
        fwd_chars = int(fwd_pct / 100 * bar_width)
        bwd_chars = int(bwd_pct / 100 * bar_width)
        opt_chars = bar_width - fwd_chars - bwd_chars

        bar = (
            f"[green]{'█' * fwd_chars}[/]"
            f"[yellow]{'█' * bwd_chars}[/]"
            f"[blue]{'█' * opt_chars}[/]"
        )

        legend = (
            f"[green]Fwd: {ms_fwd:.0f}ms ({fwd_pct:.0f}%)[/] "
            f"[yellow]Bwd: {ms_bwd:.0f}ms ({bwd_pct:.0f}%)[/] "
            f"[blue]Opt: {ms_opt:.0f}ms ({opt_pct:.0f}%)[/]"
        )

        self.update(f"{bar}\n{legend}")


class TrainingDashboard(App[None]):
    """Training dashboard with real-time metric visualization."""

    CSS = """
    TrainingDashboard {
        background: #1E1E2E;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #header-bar {
        height: 3;
        background: #2D2D3D;
        padding: 1;
    }

    #progress-section {
        height: 5;
        padding: 1;
        background: #2D2D3D;
        border-bottom: solid #45475A;
    }

    #plots-grid {
        height: 1fr;
        padding: 1;
    }

    .plot-row {
        height: 1fr;
    }

    .plot-cell {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }

    #sidebar {
        width: 35;
        background: #2D2D3D;
        border-left: solid #45475A;
        padding: 1;
    }

    #status-label {
        text-align: center;
        color: #A78BFA;
        text-style: bold;
    }

    ProgressBar {
        padding: 0 2;
    }

    ProgressBar > .bar--bar {
        color: #9ece6a;
    }

    ProgressBar > .bar--complete {
        color: #9ece6a;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "clear", "Clear"),
        Binding("p", "pause", "Pause/Resume"),
    ]

    # Reactive state
    paused: reactive[bool] = reactive(False)

    def __init__(
        self,
        log_path: Path | str | None = None,
        total_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.log_path = Path(log_path) if log_path else None
        self.metrics = TrainingMetrics()
        self.metrics.total_steps = total_steps
        self._plots: dict[str, MetricPlot] = {}
        self._last_read_pos = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            with Vertical(id="content-area"):
                # Status and progress
                with Vertical(id="progress-section"):
                    yield Label("🔬 Training Dashboard", id="status-label")
                    yield ProgressBar(total=self.metrics.total_steps or 100, id="progress-bar")

                # Plots grid (2x2)
                with Container(id="plots-grid"):
                    with Horizontal(classes="plot-row"):
                        with Container(classes="plot-cell"):
                            plot = MetricPlot("Loss", "loss", color="red", id="plot-loss")
                            self._plots["loss"] = plot
                            yield plot
                        with Container(classes="plot-cell"):
                            plot = MetricPlot("Perplexity", "ppl", color="yellow", id="plot-ppl")
                            self._plots["ppl"] = plot
                            yield plot
                    with Horizontal(classes="plot-row"):
                        with Container(classes="plot-cell"):
                            plot = MetricPlot("Learning Rate", "lr", color="cyan", id="plot-lr")
                            self._plots["lr"] = plot
                            yield plot
                        with Container(classes="plot-cell"):
                            plot = MetricPlot("Tokens/sec", "tok_s", color="green", id="plot-tok")
                            self._plots["tok_s"] = plot
                            yield plot

            # Sidebar with current metrics
            with Vertical(id="sidebar"):
                yield Label("📊 Current Metrics", id="sidebar-title")
                yield MetricsPanel(id="metrics-panel")
                yield Label("⏱ Timing Breakdown", id="timing-title")
                yield TimingBar(id="timing-bar")

        yield Footer()

    def on_mount(self) -> None:
        """Start watching log file for updates."""
        if self.log_path and self.log_path.exists():
            self.watch_log_file()

        # Initial refresh
        self.set_interval(1.0, self.refresh_display)

    @work(exclusive=True, thread=True)
    def watch_log_file(self) -> None:
        """Watch log file for new metrics."""
        if not self.log_path:
            return

        while not self.paused:
            try:
                if self.log_path.exists():
                    with open(self.log_path, "r") as f:
                        f.seek(self._last_read_pos)
                        for line in f:
                            self._parse_log_line(line.strip())
                        self._last_read_pos = f.tell()
            except Exception:
                pass

            # Poll every 500ms
            import time
            time.sleep(0.5)

    def _parse_log_line(self, line: str) -> None:
        """Parse a JSONL log line and update metrics."""
        if not line:
            return
        try:
            data = json.loads(line)
            if data.get("type") == "metrics" and "metrics" in data:
                step = data.get("step", self.metrics.current_step + 1)
                self.metrics.update(step, data["metrics"])
                self.call_from_thread(self.refresh_display)
        except json.JSONDecodeError:
            pass

    def refresh_display(self) -> None:
        """Refresh all display components."""
        m = self.metrics

        # Update progress bar
        try:
            progress = self.query_one("#progress-bar", ProgressBar)
            if m.total_steps:
                progress.total = m.total_steps
            progress.progress = m.current_step
        except Exception:
            pass

        # Update plots
        steps = list(m.steps)
        if steps:
            if "loss" in self._plots and m.loss:
                self._plots["loss"].update_plot(steps, list(m.loss))
            if "ppl" in self._plots and m.ppl:
                self._plots["ppl"].update_plot(steps, list(m.ppl))
            if "lr" in self._plots and m.lr:
                self._plots["lr"].update_plot(steps, list(m.lr))
            if "tok_s" in self._plots and m.tok_s:
                self._plots["tok_s"].update_plot(steps, list(m.tok_s))

        # Update metrics panel
        try:
            panel = self.query_one("#metrics-panel", MetricsPanel)
            panel.update_metrics(m)
        except Exception:
            pass

        # Update timing bar
        try:
            timing = self.query_one("#timing-bar", TimingBar)
            timing.update_timing(m.ms_fwd, m.ms_bwd, m.ms_opt)
        except Exception:
            pass

    def add_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Programmatically add metrics (for integration with trainer)."""
        self.metrics.update(step, metrics)
        self.refresh_display()

    def action_quit(self) -> None:
        """Quit the dashboard."""
        self.exit()

    def action_refresh(self) -> None:
        """Force refresh."""
        self.refresh_display()

    def action_clear(self) -> None:
        """Clear metrics history."""
        self.metrics = TrainingMetrics()
        self.metrics.total_steps = self.metrics.total_steps
        self.refresh_display()

    def action_pause(self) -> None:
        """Pause/resume log watching."""
        self.paused = not self.paused
        if not self.paused and self.log_path:
            self.watch_log_file()


def main(log_path: str | None = None, total_steps: int = 0) -> None:
    """Run the training dashboard."""
    app = TrainingDashboard(log_path=log_path, total_steps=total_steps)
    app.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Dashboard")
    parser.add_argument("--log", type=str, help="Path to training JSONL log file")
    parser.add_argument("--steps", type=int, default=0, help="Total training steps")
    args = parser.parse_args()
    main(log_path=args.log, total_steps=args.steps)
