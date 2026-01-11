"""Rich-based logger with caramba theming.

Training runs produce lots of output. This logger makes it readable with:
- Semantic colors (cyan=info, green=success, amber=warning, red=error)
- Structured output (tables, panels, key-value pairs)
- Progress bars and spinners
- Training-specific helpers for consistent metrics display
"""
from __future__ import annotations

from typing import Any, Generator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    track,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# Caramba color theme - distinctive and cohesive
CARAMBA_THEME = Theme(
    {
        "info": "bold #7dcfff",  # Soft cyan - informational
        "success": "bold #9ece6a",  # Muted green - success
        "warning": "bold #e0af68",  # Warm amber - warnings
        "error": "bold #f7768e",  # Soft coral red - errors
        "highlight": "bold #bb9af7",  # Lavender purple - emphasis
        "muted": "dim #565f89",  # Slate gray - secondary info
        "metric": "#7aa2f7",  # Sky blue - metrics/numbers
        "path": "italic #73daca",  # Teal - file paths
        "step": "#ff9e64",  # Orange - step/progress indicators
        "trace": "dim #91d7e3",  # Light blue - trace messages
    }
)


class Logger:
    """Unified logging interface with rich console output.

    Wraps Rich Console to provide semantic log levels, structured data
    display, and progress tracking—all with consistent theming.
    """

    def __init__(self) -> None:
        """Initialize with the caramba theme."""
        self.console = Console(theme=CARAMBA_THEME)
        self._live_display: Live | None = None
        self._live_renderable: Panel | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Basic Logging
    # ─────────────────────────────────────────────────────────────────────

    def trace(self, message: str) -> None:
        """Log a trace message (light blue ⚙)."""
        self.console.print(f"[trace]⚙[/trace] {message}")

    def log(self, message: str) -> None:
        """Log a generic message."""
        self.console.print(message)

    def info(self, message: str) -> None:
        """Log an informational message (cyan ℹ)."""
        self.console.print(f"[info]ℹ[/info] {message}")

    def success(self, message: str) -> None:
        """Log a success message (green ✓)."""
        self.console.print(f"[success]✓[/success] {message}")

    def warning(self, message: str) -> None:
        """Log a warning message (amber ⚠)."""
        self.console.print(f"[warning]⚠[/warning] {message}")

    def error(self, message: str) -> None:
        """Log an error message (red ✗)."""
        self.console.print(f"[error]✗[/error] {message}")

    # ─────────────────────────────────────────────────────────────────────
    # Structured Output
    # ─────────────────────────────────────────────────────────────────────

    def header(self, title: str, subtitle: str | None = None) -> None:
        """Print a prominent section header.

        Use this to mark major phases like "Blockwise Training" or
        "Benchmark Results".
        """
        header_text = Text()
        header_text.append("━" * 3 + " ", style="muted")
        header_text.append(title, style="highlight")
        if subtitle:
            header_text.append(f" • {subtitle}", style="muted")
        header_text.append(" " + "━" * 40, style="muted")
        self.console.print()
        self.console.print(header_text)
        self.console.print()

    def subheader(self, text: str) -> None:
        """Print a subtle subheader for subsections."""
        self.console.print(f"[muted]──[/muted] [highlight]{text}[/highlight]")

    def panel(
        self, content: str, title: str | None = None, style: str = "muted"
    ) -> None:
        """Display content in a bordered panel."""
        self.console.print(Panel(content, title=title, border_style=style))

    def fallback_warning(
        self,
        message: str,
        *,
        title: str = "PERFORMANCE FALLBACK",
    ) -> None:
        """High-visibility warning for unoptimized runtime fallbacks."""
        self.console.print(
            Panel(
                Text(message, style="warning"),
                title=title,
                title_align="left",
                border_style="warning",
            )
        )

    def log_decision(self, from_strat: str, to_strat: str, reason: Any) -> None:
        """Render an orchestrator switch decision as a Rich panel."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="muted")
        table.add_column("Value", style="metric")
        table.add_row("from", str(from_strat))
        table.add_row("to", str(to_strat))

        if reason is not None:
            table.add_row("trigger_metric", str(getattr(reason, "trigger_metric", "")))
            table.add_row("current_value", str(getattr(reason, "current_value", "")))
            table.add_row("threshold", str(getattr(reason, "threshold", "")))
            window = getattr(reason, "window", None)
            horizon = getattr(reason, "horizon", None)
            if window is not None:
                table.add_row("window", str(window))
            if horizon is not None:
                table.add_row("horizon", str(horizon))

        self.console.print(
            Panel(
                table,
                title="ORCHESTRATOR SWITCH",
                title_align="left",
                border_style="highlight",
            )
        )

    def table(
        self,
        title: str | None = None,
        columns: list[str] | None = None,
        rows: list[list[str]] | None = None,
    ) -> Table:
        """Create and optionally populate a styled table.

        If columns and rows are provided, prints immediately. Otherwise
        returns the Table for manual population.
        """
        table = Table(
            title=title,
            title_style="highlight",
            header_style="info",
            border_style="muted",
            row_styles=["", "dim"],
        )

        if columns and rows:
            for col in columns:
                table.add_column(col)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)

        return table

    def key_value(self, data: dict[str, Any], title: str | None = None) -> None:
        """Display key-value pairs in a clean format."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="muted")
        table.add_column("Value", style="metric")

        for key, value in data.items():
            table.add_row(f"{key}:", str(value))

        if title:
            self.subheader(title)
        self.console.print(table)

    def metric(self, name: str, value: float | int | str, unit: str = "") -> None:
        """Display a single metric with formatting.

        Floats are formatted to 4 decimal places.
        """
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        self.console.print(
            f"  [muted]{name}:[/muted] [metric]{formatted}[/metric]{unit}"
        )

    def step(self, current: int, total: int | None = None, message: str = "") -> None:
        """Display a step indicator for multi-phase operations."""
        if total:
            prefix = f"[step][{current}/{total}][/step]"
        else:
            prefix = f"[step][{current}][/step]"
        self.console.print(f"{prefix} {message}")

    def path(self, filepath: str, label: str = "") -> None:
        """Display a file path with optional label."""
        if label:
            self.console.print(f"  [muted]{label}:[/muted] [path]{filepath}[/path]")
        else:
            self.console.print(f"  [path]{filepath}[/path]")

    # ─────────────────────────────────────────────────────────────────────
    # Inspection
    # ─────────────────────────────────────────────────────────────────────

    def inspect(self, obj: object, **kwargs: Any) -> None:
        """Print an object with rich formatting.

        Delegates to console.print() which handles dicts, lists, etc.
        """
        self.console.print(obj, **kwargs)

    # ─────────────────────────────────────────────────────────────────────
    # Progress Tracking
    # ─────────────────────────────────────────────────────────────────────

    def progress(self, total: int, description: str) -> Generator[int, None, None]:
        """Track iteration progress with a styled progress bar."""
        yield from track(
            range(total),
            description=f"[info]{description}[/info]",
            console=self.console,
        )

    def spinner(self, description: str = "Processing...") -> Progress:
        """Create a spinner for indeterminate progress.

        Usage:
            with logger.spinner("Loading model...") as progress:
                task = progress.add_task("", total=None)
                # ... do work ...
        """
        return Progress(
            SpinnerColumn(style="info"),
            TextColumn("[info]{task.description}[/info]"),
            console=self.console,
            transient=True,
        )

    def progress_bar(self) -> Progress:
        """Create a rich progress bar for fine-grained control.

        Usage:
            with logger.progress_bar() as progress:
                task = progress.add_task("Training...", total=1000)
                for step in range(1000):
                    progress.update(task, advance=1)
        """
        return Progress(
            SpinnerColumn(style="info"),
            TextColumn("[info]{task.description}[/info]"),
            BarColumn(bar_width=40, style="muted", complete_style="success"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Training-Specific Helpers
    # ─────────────────────────────────────────────────────────────────────

    def training_step(
        self,
        phase: str,
        step: int,
        loss: float,
        extras: dict[str, float] | None = None,
    ) -> None:
        """Log a training step with consistent formatting.

        Args:
            phase: Training phase (e.g., "blockwise", "global")
            step: Current step number
            loss: Loss value
            extras: Additional metrics like {"ce": 0.5, "diff": 0.1}
        """
        parts = [
            f"[step]{phase}[/step] step=[metric]{step}[/metric] "
            f"loss=[metric]{loss:.6f}[/metric]"
        ]
        if extras:
            extra_str = " ".join(
                f"{k}=[metric]{v:.4f}[/metric]" for k, v in extras.items()
            )
            parts.append(f"({extra_str})")
        self.console.print(" ".join(parts))

    def benchmark_result(
        self,
        name: str,
        model: str,
        value: float,
        unit: str = "",
    ) -> None:
        """Log a benchmark result with consistent formatting."""
        self.console.print(
            f"  [muted]{model}:[/muted] [metric]{value:.2f}[/metric]{unit}"
        )

    def tuner_status(self, metrics: dict[str, Any]) -> None:
        """Lightweight visualization of auto-tuning status.
        
        Args:
            metrics: Dict containing 'actual', 'target', and 'velocity' for levers.
        """
        table = Table(
            show_header=True, 
            header_style="info", 
            border_style="muted",
            box=None,
            padding=(0, 2)
        )
        table.add_column("Parameter", style="muted", width=25)
        table.add_column("Value", justify="right", style="metric", width=12)
        table.add_column("Momentum", justify="right", width=15)

        for name, data in metrics.items():
            actual = data.get("actual", 0.0)
            velocity = data.get("velocity", 0.0)
            
            # Choose color based on velocity direction
            if velocity > 0:
                speed_style = "success"
                arrow = "↑"
            elif velocity < 0:
                speed_style = "amber"
                arrow = "↓"
            else:
                speed_style = "muted"
                arrow = "•"

            # Format values
            actual_str = f"{actual:.3f}" if isinstance(actual, float) else str(actual)
            speed_str = f"[{speed_style}]{arrow} {abs(velocity):.4f}[/{speed_style}]"
            
            table.add_row(name, actual_str, speed_str)

        panel = Panel(
            table,
            title="MEM TUNER",
            title_align="left",
            border_style="muted",
            padding=(0, 1),
            expand=False
        )
        
        # Store tuner panel separately
        self._tuner_panel = panel
        
        # Auto-start Live display on first call if not already active
        if self._live_display is None:
            self._live_display = Live(
                panel,
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self._live_display.start()
        
        # Update display with both panels if health exists
        if hasattr(self, '_health_panel') and self._health_panel is not None:
            from rich.console import Group
            combined = Group(self._tuner_panel, self._health_panel)
            self._live_display.update(combined)
        else:
            self._live_display.update(panel)
    
    def health_bars(self, metrics: dict[str, float]) -> None:
        """Display health metrics as color-coded bars.
        
        Args:
            metrics: Dict with 'accuracy', 'loss_variance', 'utilization', 'objective'
        """
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        
        table = Table(
            show_header=False,
            border_style="muted",
            box=None,
            padding=(0, 1),
            expand=False
        )
        table.add_column("Metric", style="muted", width=18)
        table.add_column("Bar", width=40)
        table.add_column("Value", justify="right", width=10)
        
        def make_bar(value: float, max_val: float, reverse: bool = False) -> Text:
            """Create a color-coded health bar.
            
            Args:
                value: Current value
                max_val: Maximum value for scaling (defines the '100% range')
                reverse: If True, low value = high health (e.g., loss)
            """
            # Normalize to 0.0 - 1.0 range based on max_val
            norm_val = max(0.0, min(1.0, value / max_val))
            
            # Determine "Health Percentage" (0.0 = Dead/Red, 1.0 = Full/Green)
            if reverse:
                # For loss: 0 is 100% healthy, max_val is 0% healthy
                health_pct = 1.0 - norm_val
            else:
                # For accuracy: 1.0 is 100% healthy
                health_pct = norm_val

            # Determine color based on health
            if health_pct > 0.8:
                color = "bold green"     # Excellent
            elif health_pct > 0.5:
                color = "green"          # Good
            elif health_pct > 0.3:
                color = "yellow"         # Warning
            else:
                color = "red"            # Critical
                
            bar_width = 25
            filled_len = int(health_pct * bar_width)
            empty_len = bar_width - filled_len
            
            # Create composite text: Colored filled part + Dim empty part
            bar_text = Text()
            bar_text.append("█" * filled_len, style=color)
            bar_text.append("░" * empty_len, style="dim #444444")
            return bar_text
        
        # Accuracy (0-1 scale)
        acc = metrics.get("accuracy", 0.0)
        table.add_row(
            "Accuracy",
            make_bar(acc, 1.0, reverse=False),
            f"{acc:.3f}"
        )
        
        # Loss Variance (0-10 scale, lower is better)
        loss_var = metrics.get("loss_variance", 0.0)
        table.add_row(
            "Loss Stability",
            make_bar(loss_var, 10.0, reverse=True),
            f"{loss_var:.3f}"
        )
        
        # Utilization (0-1 scale, ~0.5 is ideal)
        util = metrics.get("utilization", 0.0)
        # Map to health: 0.5 = perfect, further away = worse
        util_health = 1.0 - abs(util - 0.5) * 2.0
        table.add_row(
            "Memory Usage",
            make_bar(util_health, 1.0, reverse=False),
            f"{util:.3f}"
        )
        
        # Objective (normalize to 0-100 scale)
        obj = metrics.get("objective", 0.0)
        obj_normalized = max(0.0, min(1.0, obj / 100.0))
        table.add_row(
            "Overall Health",
            make_bar(obj_normalized, 1.0, reverse=False),
            f"{obj:.1f}"
        )
        
        panel = Panel(
            table,
            title="SYSTEM HEALTH",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            expand=False
        )
        
        # Store health panel separately
        self._health_panel = panel
        
        # Update Live display with both panels if tuner exists
        if self._live_display is not None and hasattr(self, '_tuner_panel') and self._tuner_panel is not None:
            from rich.console import Group
            combined = Group(self._tuner_panel, self._health_panel)
            self._live_display.update(combined)
        elif self._live_display is not None:
            self._live_display.update(panel)
        else:
            # Print directly if Live not started yet
            self.console.print(panel)

    @contextmanager
    def live_display(self):
        """Context manager for Live display updates.
        
        Usage:
            with logger.live_display():
                # Calls to tuner_status() will update in place
                logger.tuner_status(metrics)
        """
        if self._live_display is not None:
            # Already in a live display context, just yield
            yield
            return
            
        self._live_display = Live(
            "",
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        try:
            self._live_display.start()
            yield
        finally:
            if self._live_display is not None:
                self._live_display.stop()
            self._live_display = None
            self._live_renderable = None

    def artifacts_summary(self, artifacts: dict[str, Any]) -> None:
        """Display a summary of generated artifacts."""
        self.console.print()
        self.success(f"Generated {len(artifacts)} artifacts:")
        for name, path in artifacts.items():
            self.console.print(f"    [muted]•[/muted] {name}: [path]{path}[/path]")


# ─────────────────────────────────────────────────────────────────────────────
# Module-Level Singleton
# ─────────────────────────────────────────────────────────────────────────────

_logger: Logger | None = None


def get_logger() -> Logger:
    """Get or create the singleton Logger instance.

    Using a singleton ensures consistent theming and avoids creating
    multiple Console instances.
    """
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger
