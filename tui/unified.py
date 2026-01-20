"""Unified Caramba TUI with multiple views.

A single TUI entrypoint that allows switching between:
- Chat: Agent chat interface
- Training: Real-time training metrics dashboard
- Builder: Visual manifest/architecture builder

Switch views with Ctrl+1/2/3 or via the command palette.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    ProgressBar,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

import yaml

# Import existing components
from tui.styles import TUI_CSS
from tui.viewport import Viewport
from tui.input_bar import InputBar
from tui.sidebars import (
    ExpertsSidebar,
    ToolsSidebar,
    StatusBar,
    ExpertStatus,
    AgentStatus,
    AgentDetailModal,
    ToolDetailModal,
)
from tui.command_palette import CommandPalette, HelpScreen
from tui.commands import Command

# Import training dashboard components
from tui.training_dashboard import TrainingMetrics, MetricPlot, MetricsPanel, TimingBar

# Import manifest builder components
from tui.manifest_builder import (
    ManifestConfig,
    LayerConfig,
    LayerListItem,
    LayerConfigPanel,
    TopologyView,
    LAYER_TYPES,
    LAYER_CONFIGS,
)

# Check for plotext
try:
    from textual_plotext import PlotextPlot
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    PlotextPlot = None  # type: ignore[misc,assignment]


def get_base_url(url: str) -> str:
    """Extract the base URL (scheme + host + port) from a full URL."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Chat View
# ─────────────────────────────────────────────────────────────────────────────

class ChatView(Vertical):
    """Chat interface view."""

    DEFAULT_CSS = """
    ChatView {
        width: 100%;
        height: 100%;
    }

    ChatView #chat-main {
        width: 100%;
        height: 1fr;
    }

    ChatView #left-sidebar {
        width: 25;
        background: #2D2D3D;
        border-right: solid #45475A;
    }

    ChatView #chat-area {
        width: 1fr;
    }

    ChatView #right-sidebar {
        width: 30;
        background: #2D2D3D;
        border-left: solid #45475A;
    }
    """

    def __init__(self, root_agent_url: str = "http://localhost:9000", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root_agent_url = root_agent_url
        self._viewport: Viewport | None = None
        self._experts_sidebar: ExpertsSidebar | None = None
        self._tools_sidebar: ToolsSidebar | None = None
        self._input_bar: InputBar | None = None
        self._is_streaming = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat-main"):
            with Vertical(id="left-sidebar"):
                yield ExpertsSidebar(id="experts-sidebar")
            with Vertical(id="chat-area"):
                yield Viewport(id="viewport")
                yield InputBar(id="input-bar")
            with Vertical(id="right-sidebar"):
                yield ToolsSidebar(id="tools-sidebar")

    def on_mount(self) -> None:
        """Initialize chat components."""
        self._viewport = self.query_one("#viewport", Viewport)
        self._experts_sidebar = self.query_one("#experts-sidebar", ExpertsSidebar)
        self._tools_sidebar = self.query_one("#tools-sidebar", ToolsSidebar)
        self._input_bar = self.query_one("#input-bar", InputBar)

    def focus_input(self) -> None:
        """Focus the input bar."""
        if self._input_bar:
            self._input_bar.focus_input()


# ─────────────────────────────────────────────────────────────────────────────
# Training View
# ─────────────────────────────────────────────────────────────────────────────

class TrainingView(Vertical):
    """Training dashboard view with real-time plots."""

    DEFAULT_CSS = """
    TrainingView {
        width: 100%;
        height: 100%;
    }

    TrainingView #training-main {
        width: 100%;
        height: 1fr;
    }

    TrainingView #training-content {
        width: 1fr;
    }

    TrainingView #progress-section {
        height: 5;
        padding: 1;
        background: #2D2D3D;
        border-bottom: solid #45475A;
    }

    TrainingView #plots-grid {
        height: 1fr;
        padding: 1;
    }

    TrainingView .plot-row {
        height: 1fr;
    }

    TrainingView .plot-cell {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }

    TrainingView #training-sidebar {
        width: 35;
        background: #2D2D3D;
        border-left: solid #45475A;
        padding: 1;
    }

    TrainingView #status-label {
        text-align: center;
        color: #A78BFA;
        text-style: bold;
        padding: 1;
    }
    """

    def __init__(self, log_path: Path | str | None = None, total_steps: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log_path = Path(log_path) if log_path else None
        self.metrics = TrainingMetrics()
        self.metrics.total_steps = total_steps
        self._plots: dict[str, MetricPlot] = {}
        self._last_read_pos = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="training-main"):
            with Vertical(id="training-content"):
                with Vertical(id="progress-section"):
                    yield Label("🔬 Training Dashboard", id="status-label")
                    yield ProgressBar(total=self.metrics.total_steps or 100, id="progress-bar")

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

            with Vertical(id="training-sidebar"):
                yield Label("📊 Current Metrics", id="sidebar-title")
                yield MetricsPanel(id="metrics-panel")
                yield Label("⏱ Timing Breakdown", id="timing-title")
                yield TimingBar(id="timing-bar")

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
        """Programmatically add metrics."""
        self.metrics.update(step, metrics)
        self.refresh_display()


# ─────────────────────────────────────────────────────────────────────────────
# Builder View
# ─────────────────────────────────────────────────────────────────────────────

class BuilderView(Vertical):
    """Manifest builder view."""

    DEFAULT_CSS = """
    BuilderView {
        width: 100%;
        height: 100%;
    }

    BuilderView #builder-main {
        width: 100%;
        height: 1fr;
    }

    BuilderView #builder-left {
        width: 35;
        background: #2D2D3D;
        border-right: solid #45475A;
    }

    BuilderView #builder-center {
        width: 1fr;
        padding: 1;
    }

    BuilderView #builder-right {
        width: 45;
        background: #2D2D3D;
        border-left: solid #45475A;
        padding: 1;
    }

    BuilderView #layer-list {
        height: 1fr;
    }

    BuilderView #add-layer-section {
        height: auto;
        padding: 1;
        border-top: solid #45475A;
    }

    BuilderView .section-title {
        text-style: bold;
        color: #A78BFA;
        padding: 1;
    }

    BuilderView Button {
        margin: 1;
    }
    """

    manifest: reactive[ManifestConfig] = reactive(ManifestConfig, recompose=False)
    selected_layer_index: reactive[int] = reactive(-1)

    def __init__(self, output_path: Path | str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_path = Path(output_path) if output_path else Path("manifest.yml")
        self.manifest = ManifestConfig()

    def compose(self) -> ComposeResult:
        with Horizontal(id="builder-main"):
            # Left panel: Layer list
            with Vertical(id="builder-left"):
                yield Label("📚 Layers", classes="section-title")
                yield ListView(id="layer-list")

                with Vertical(id="add-layer-section"):
                    yield Label("Add Layer", classes="section-title")
                    yield Select(
                        options=[(name, key) for key, name in LAYER_TYPES],
                        id="layer-type-select",
                        prompt="Select layer type",
                    )
                    yield Input(placeholder="Layer name", id="layer-name-input")
                    yield Button("Add Layer", id="add-layer-btn", variant="primary")

            # Center panel: Config + Topology
            with Vertical(id="builder-center"):
                with TabbedContent():
                    with TabPane("Configuration", id="config-tab"):
                        yield VerticalScroll(LayerConfigPanel(id="layer-config-panel"))

                    with TabPane("Topology", id="topology-tab"):
                        yield TopologyView(id="topology-view")

                    with TabPane("Training", id="training-tab"):
                        yield self._build_training_config()

            # Right panel: YAML preview
            with Vertical(id="builder-right"):
                yield Label("📄 YAML Preview", classes="section-title")
                yield TextArea(
                    self.manifest.to_yaml(),
                    language="yaml",
                    read_only=True,
                    id="yaml-preview",
                )
                yield Button("💾 Save Manifest", id="save-btn", variant="success")

    def _build_training_config(self) -> Vertical:
        """Build training configuration panel."""
        return Vertical(
            Label("Training Configuration", classes="section-title"),
            Horizontal(
                Label("Batch Size:", classes="config-label"),
                Input(value=str(self.manifest.batch_size), type="integer", id="train-batch-size"),
                classes="config-row",
            ),
            Horizontal(
                Label("Block Size:", classes="config-label"),
                Input(value=str(self.manifest.block_size), type="integer", id="train-block-size"),
                classes="config-row",
            ),
            Horizontal(
                Label("Learning Rate:", classes="config-label"),
                Input(value=str(self.manifest.lr), type="number", id="train-lr"),
                classes="config-row",
            ),
            Horizontal(
                Label("Steps:", classes="config-label"),
                Input(value=str(self.manifest.steps), type="integer", id="train-steps"),
                classes="config-row",
            ),
            Horizontal(
                Label("Device:", classes="config-label"),
                Select(
                    options=[("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"), ("MPS", "mps")],
                    value=self.manifest.device,
                    id="train-device",
                ),
                classes="config-row",
            ),
        )

    def on_mount(self) -> None:
        """Initialize builder."""
        self._refresh_layer_list()
        self._refresh_preview()

    def _refresh_layer_list(self) -> None:
        """Refresh the layer list."""
        try:
            list_view = self.query_one("#layer-list", ListView)
            list_view.clear()
            for i, layer in enumerate(self.manifest.layers):
                list_view.append(LayerListItem(layer, i))
        except Exception:
            pass

    def _refresh_preview(self) -> None:
        """Refresh YAML preview."""
        try:
            preview = self.query_one("#yaml-preview", TextArea)
            preview.load_text(self.manifest.to_yaml())
        except Exception:
            pass

        try:
            topology = self.query_one("#topology-view", TopologyView)
            topology.render_topology(self.manifest.layers)
        except Exception:
            pass

    @on(Button.Pressed, "#add-layer-btn")
    def add_layer(self) -> None:
        """Add a new layer."""
        try:
            layer_type = self.query_one("#layer-type-select", Select).value
            layer_name = self.query_one("#layer-name-input", Input).value

            if not layer_type or not layer_name:
                return

            defaults = {}
            for key, _, _, default in LAYER_CONFIGS.get(str(layer_type), []):
                defaults[key] = default[0] if isinstance(default, list) else default

            layer = LayerConfig(layer_type=str(layer_type), name=layer_name, config=defaults)
            self.manifest.layers.append(layer)

            self.query_one("#layer-name-input", Input).value = ""
            self._refresh_layer_list()
            self._refresh_preview()
        except Exception:
            pass

    @on(Button.Pressed, "#save-btn")
    def save_manifest(self) -> None:
        """Save manifest to file."""
        self.output_path.write_text(self.manifest.to_yaml())
        self.app.notify(f"Saved to {self.output_path}", title="Manifest Saved")


# ─────────────────────────────────────────────────────────────────────────────
# Unified App
# ─────────────────────────────────────────────────────────────────────────────

class CarambaApp(App[None]):
    """Unified Caramba TUI with multiple views."""

    CSS = TUI_CSS + """
    #view-tabs {
        height: 100%;
    }

    .view-container {
        height: 100%;
    }

    #view-indicator {
        dock: top;
        height: 1;
        background: #2D2D3D;
        padding: 0 2;
    }

    #view-indicator Label {
        margin-right: 3;
    }

    .view-tab {
        padding: 0 2;
    }

    .view-tab-active {
        background: #45475A;
        color: #A78BFA;
        text-style: bold;
    }
    """

    TITLE = "Caramba"
    SUB_TITLE = "AI Research Platform"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+1", "view_chat", "Chat", priority=True),
        Binding("ctrl+2", "view_training", "Training", priority=True),
        Binding("ctrl+3", "view_builder", "Builder", priority=True),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("f1", "help", "Help"),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "focus_next", "Next", show=False),
    ]

    current_view: reactive[str] = reactive("chat")

    def __init__(
        self,
        root_agent_url: str = "http://localhost:9000",
        log_path: Path | str | None = None,
        total_steps: int = 0,
        manifest_output: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root_agent_url = root_agent_url
        self.log_path = log_path
        self.total_steps = total_steps
        self.manifest_output = manifest_output

        self._chat_view: ChatView | None = None
        self._training_view: TrainingView | None = None
        self._builder_view: BuilderView | None = None
        self._status_bar: StatusBar | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        # View indicator tabs
        with Horizontal(id="view-indicator"):
            yield Label("[bold cyan]Chat[/] (Ctrl+1)", id="tab-chat", classes="view-tab view-tab-active")
            yield Label("Training (Ctrl+2)", id="tab-training", classes="view-tab")
            yield Label("Builder (Ctrl+3)", id="tab-builder", classes="view-tab")

        # Content switcher for views
        with ContentSwitcher(initial="chat", id="view-switcher"):
            with Vertical(id="chat", classes="view-container"):
                yield ChatView(root_agent_url=self.root_agent_url, id="chat-view")

            with Vertical(id="training", classes="view-container"):
                yield TrainingView(
                    log_path=self.log_path,
                    total_steps=self.total_steps,
                    id="training-view",
                )

            with Vertical(id="builder", classes="view-container"):
                yield BuilderView(output_path=self.manifest_output, id="builder-view")

        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app."""
        self._chat_view = self.query_one("#chat-view", ChatView)
        self._training_view = self.query_one("#training-view", TrainingView)
        self._builder_view = self.query_one("#builder-view", BuilderView)
        self._status_bar = self.query_one("#status-bar", StatusBar)

        # Set initial status
        self._status_bar.set_agent_url(self.root_agent_url)

        # Focus chat input
        if self._chat_view:
            self._chat_view.focus_input()

        # Start agent health check
        self.refresh_agent_hierarchy()
        self.set_interval(10.0, self.refresh_agent_hierarchy)

        # Start training log watcher if path provided
        if self.log_path:
            self.watch_training_log()

    def watch_view(self, view: str) -> None:
        """React to view changes."""
        self._update_tab_indicators()
        switcher = self.query_one("#view-switcher", ContentSwitcher)
        switcher.current = view

    def _update_tab_indicators(self) -> None:
        """Update tab styling based on current view."""
        tabs = {
            "chat": self.query_one("#tab-chat", Label),
            "training": self.query_one("#tab-training", Label),
            "builder": self.query_one("#tab-builder", Label),
        }

        for name, tab in tabs.items():
            tab.remove_class("view-tab-active")
            if name == self.current_view:
                tab.add_class("view-tab-active")
                if name == "chat":
                    tab.update("[bold cyan]Chat[/] (Ctrl+1)")
                elif name == "training":
                    tab.update("[bold cyan]Training[/] (Ctrl+2)")
                elif name == "builder":
                    tab.update("[bold cyan]Builder[/] (Ctrl+3)")
            else:
                if name == "chat":
                    tab.update("Chat (Ctrl+1)")
                elif name == "training":
                    tab.update("Training (Ctrl+2)")
                elif name == "builder":
                    tab.update("Builder (Ctrl+3)")

    def action_view_chat(self) -> None:
        """Switch to chat view."""
        self.current_view = "chat"
        self._update_tab_indicators()
        switcher = self.query_one("#view-switcher", ContentSwitcher)
        switcher.current = "chat"
        if self._chat_view:
            self._chat_view.focus_input()

    def action_view_training(self) -> None:
        """Switch to training view."""
        self.current_view = "training"
        self._update_tab_indicators()
        switcher = self.query_one("#view-switcher", ContentSwitcher)
        switcher.current = "training"

    def action_view_builder(self) -> None:
        """Switch to builder view."""
        self.current_view = "builder"
        self._update_tab_indicators()
        switcher = self.query_one("#view-switcher", ContentSwitcher)
        switcher.current = "builder"

    def action_quit(self) -> None:
        """Quit the app."""
        self.exit()

    def action_command_palette(self) -> None:
        """Open command palette."""
        self.push_screen(CommandPalette())

    def action_help(self) -> None:
        """Show help."""
        self.push_screen(HelpScreen())

    def action_cancel(self) -> None:
        """Cancel/close modals."""
        try:
            modal = self.query_one(ToolDetailModal)
            modal.remove()
            return
        except Exception:
            pass
        try:
            modal = self.query_one(AgentDetailModal)
            modal.remove()
            return
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Agent communication (delegated from ChatView)
    # ─────────────────────────────────────────────────────────────────────────

    @work(exclusive=False)
    async def refresh_agent_hierarchy(self) -> None:
        """Fetch agent hierarchy."""
        if not self._chat_view or not self._chat_view._experts_sidebar:
            return

        base_url = get_base_url(self.root_agent_url)
        sidebar = self._chat_view._experts_sidebar

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 404:
                        response = await client.get(f"{base_url}/.well-known/agent-card.json")
                    root_healthy = response.status_code == 200
                    if self._status_bar:
                        self._status_bar.set_connected(root_healthy)
                except Exception:
                    root_healthy = False
                    if self._status_bar:
                        self._status_bar.set_connected(False)

                # Get hierarchy
                try:
                    response = await client.get(f"{base_url}/agents/status", timeout=15.0)
                    if response.status_code == 200:
                        data = response.json()
                        root_info = data.get("root", {})
                        root_name = root_info.get("name", "Root")
                        teams = data.get("teams", {})
                        sub_agents = data.get("sub_agents", {})

                        if isinstance(teams, dict) and teams:
                            sidebar.set_root_teams(
                                root_name,
                                AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                                teams,
                            )
                            for _, tinfo in teams.items():
                                agents = tinfo.get("agents", {}) if isinstance(tinfo, dict) else {}
                                if not isinstance(agents, dict):
                                    continue
                                for agent_name, info in agents.items():
                                    if not isinstance(info, dict):
                                        continue
                                    healthy = info.get("healthy", False)
                                    error = info.get("error", "")
                                    sidebar.update_sub_agent_status(
                                        agent_name,
                                        AgentStatus.HEALTHY if healthy else AgentStatus.UNHEALTHY,
                                        error if not healthy else "",
                                    )
                            return
                        lead_children: dict[str, list[str]] = {}
                        for lead_name, info in sub_agents.items():
                            sub = info.get("sub_agents", {}) if isinstance(info, dict) else {}
                            if isinstance(sub, dict) and sub:
                                lead_children[lead_name] = list(sub.keys())

                        sidebar.set_root_agent(
                            root_name,
                            AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                            list(sub_agents.keys()),
                            lead_children if lead_children else None,
                        )

                        for name, info in sub_agents.items():
                            healthy = info.get("healthy", False)
                            error = info.get("error", "")
                            sidebar.update_sub_agent_status(
                                name,
                                AgentStatus.HEALTHY if healthy else AgentStatus.UNHEALTHY,
                                error if not healthy else "",
                            )
                            nested = info.get("sub_agents", {}) if isinstance(info, dict) else {}
                            if isinstance(nested, dict) and nested:
                                for member_name, member_info in nested.items():
                                    if not isinstance(member_info, dict):
                                        continue
                                    m_ok = member_info.get("healthy", False)
                                    m_err = member_info.get("error", "")
                                    sidebar.update_sub_agent_status(
                                        member_name,
                                        AgentStatus.HEALTHY if m_ok else AgentStatus.UNHEALTHY,
                                        m_err if not m_ok else "",
                                    )
                        return
                except Exception:
                    pass

                sidebar.set_root_agent(
                    "Root",
                    AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                )

        except Exception:
            sidebar.set_root_agent("Root", AgentStatus.UNHEALTHY, message="connection failed")

    # ─────────────────────────────────────────────────────────────────────────
    # Training log watching
    # ─────────────────────────────────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def watch_training_log(self) -> None:
        """Watch training log file for updates."""
        if not self.log_path or not self._training_view:
            return

        log_path = Path(self.log_path)
        last_pos = 0

        import time
        while True:
            try:
                if log_path.exists():
                    with open(log_path) as f:
                        f.seek(last_pos)
                        for line in f:
                            self._parse_training_log_line(line.strip())
                        last_pos = f.tell()
            except Exception:
                pass

            time.sleep(0.5)

    def _parse_training_log_line(self, line: str) -> None:
        """Parse a training log line."""
        if not line or not self._training_view:
            return
        try:
            data = json.loads(line)
            if data.get("type") == "metrics" and "metrics" in data:
                step = data.get("step", 0)
                self._training_view.metrics.update(step, data["metrics"])
                self.call_from_thread(self._training_view.refresh_display)
        except json.JSONDecodeError:
            pass


def main() -> None:
    """Main entrypoint for unified TUI."""
    import argparse

    parser = argparse.ArgumentParser(description="Caramba TUI")
    parser.add_argument("--url", type=str, default="http://localhost:9000", help="Root agent URL")
    parser.add_argument("--log", type=str, default=None, help="Training log path")
    parser.add_argument("--steps", type=int, default=0, help="Total training steps")
    parser.add_argument("--output", type=str, default="manifest.yml", help="Manifest output path")
    args = parser.parse_args()

    app = CarambaApp(
        root_agent_url=args.url,
        log_path=args.log,
        total_steps=args.steps,
        manifest_output=args.output,
    )
    app.run()


if __name__ == "__main__":
    main()
