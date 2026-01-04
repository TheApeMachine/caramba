"""Manifest Builder TUI for visual architecture design.

This module provides manifest builder components that can be used standalone
or as part of the unified TUI (see unified.py).

Components:
- ManifestConfig: Configuration dataclass for full manifests
- LayerConfig: Configuration for a single layer
- LayerListItem: ListView item for layers
- LayerConfigPanel: Configuration panel for layer parameters
- TopologyView: ASCII topology visualization
- ManifestBuilder: Standalone builder app (prefer unified.py)

For the unified TUI with all views, use:
    caramba tui

Build manifests interactively instead of writing YAML by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option


# Available layer types
LAYER_TYPES = [
    ("attention", "Attention Layer"),
    ("mosaic", "MOSAIC Block"),
    ("ffn", "Feed-Forward Network"),
    ("embedding", "Embedding Layer"),
    ("norm", "Normalization"),
    ("output", "Output Head"),
]

# Layer configurations
LAYER_CONFIGS: dict[str, list[tuple[str, str, str, Any]]] = {
    "attention": [
        ("n_heads", "Number of Heads", "int", 8),
        ("d_model", "Model Dimension", "int", 512),
        ("d_head", "Head Dimension", "int", 64),
        ("dropout", "Dropout", "float", 0.1),
        ("mode", "Attention Mode", "select", ["full", "local", "global", "hybrid"]),
        ("rope", "Use RoPE", "bool", True),
        ("flash", "Use Flash Attention", "bool", True),
    ],
    "mosaic": [
        ("d_model", "Model Dimension", "int", 512),
        ("n_heads", "Number of Heads", "int", 8),
        ("mem_buckets", "Memory Buckets", "int", 64),
        ("mem_hashes", "Memory Hashes", "int", 4),
        ("mem_topk", "Memory Top-K", "int", 8),
        ("dropout", "Dropout", "float", 0.1),
        ("use_memory", "Enable Memory", "bool", True),
    ],
    "ffn": [
        ("d_model", "Model Dimension", "int", 512),
        ("d_ff", "FFN Dimension", "int", 2048),
        ("activation", "Activation", "select", ["gelu", "relu", "swish", "silu"]),
        ("dropout", "Dropout", "float", 0.1),
        ("gated", "Use Gated FFN", "bool", False),
    ],
    "embedding": [
        ("vocab_size", "Vocabulary Size", "int", 50257),
        ("d_model", "Model Dimension", "int", 512),
        ("max_seq_len", "Max Sequence Length", "int", 2048),
        ("dropout", "Dropout", "float", 0.1),
        ("learned_pos", "Learned Positional Embedding", "bool", False),
    ],
    "norm": [
        ("d_model", "Model Dimension", "int", 512),
        ("norm_type", "Normalization Type", "select", ["layernorm", "rmsnorm", "groupnorm"]),
        ("eps", "Epsilon", "float", 1e-5),
    ],
    "output": [
        ("d_model", "Model Dimension", "int", 512),
        ("vocab_size", "Vocabulary Size", "int", 50257),
        ("tie_weights", "Tie with Embedding", "bool", True),
    ],
}


@dataclass
class LayerConfig:
    """Configuration for a single layer."""

    layer_type: str
    name: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to manifest-compatible dict."""
        return {
            "type": self.layer_type,
            "name": self.name,
            "config": self.config,
        }


@dataclass
class ManifestConfig:
    """Full manifest configuration."""

    name: str = "my_model"
    version: str = "2"
    layers: list[LayerConfig] = field(default_factory=list)

    # Training config
    batch_size: int = 32
    block_size: int = 1024
    lr: float = 3e-4
    steps: int = 10000
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Convert to full manifest dict."""
        return {
            "version": self.version,
            "name": self.name,
            "targets": [
                {
                    "type": "experiment",
                    "name": self.name,
                    "system": {
                        "type": "transformer",
                        "config": {
                            "layers": [l.to_dict() for l in self.layers],
                        },
                    },
                    "runs": [
                        {
                            "id": "train",
                            "steps": self.steps,
                            "train": {
                                "device": self.device,
                                "batch_size": self.batch_size,
                                "block_size": self.block_size,
                                "lr": self.lr,
                            },
                        }
                    ],
                }
            ],
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class LayerListItem(ListItem):
    """A layer item in the layers list."""

    def __init__(self, layer: LayerConfig, index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._layer = layer
        self._index = index

    @property
    def layer(self) -> LayerConfig:
        return self._layer

    @property
    def index(self) -> int:
        return self._index

    def compose(self) -> ComposeResult:
        icon = {
            "attention": "🔍",
            "mosaic": "🧩",
            "ffn": "⚡",
            "embedding": "📝",
            "norm": "📐",
            "output": "🎯",
        }.get(self._layer.layer_type, "📦")
        yield Label(f"{icon} {self._layer.name} ({self._layer.layer_type})")


class LayerConfigPanel(Vertical):
    """Panel for configuring a layer."""

    DEFAULT_CSS = """
    LayerConfigPanel {
        width: 100%;
        height: auto;
        padding: 1;
        background: #2D2D3D;
        border: round #45475A;
    }

    LayerConfigPanel .config-row {
        height: 3;
        margin-bottom: 1;
    }

    LayerConfigPanel .config-label {
        width: 25;
        padding-top: 1;
    }

    LayerConfigPanel Input {
        width: 1fr;
    }

    LayerConfigPanel Select {
        width: 1fr;
    }
    """

    class ConfigChanged(Message):
        """Sent when configuration changes."""

        def __init__(self, layer_index: int, key: str, value: Any) -> None:
            super().__init__()
            self.layer_index = layer_index
            self.key = key
            self.value = value

    def __init__(self, layer: LayerConfig | None = None, index: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._layer = layer
        self._index = index

    @property
    def layer(self) -> LayerConfig | None:
        return self._layer

    @property
    def index(self) -> int:
        return self._index

    def compose(self) -> ComposeResult:
        if self._layer is None:
            yield Label("[dim]Select a layer to configure[/]")
            return

        yield Label(f"[bold]{self._layer.name}[/] ({self._layer.layer_type})", id="layer-title")

        configs = LAYER_CONFIGS.get(self._layer.layer_type, [])
        for key, label, config_type, default in configs:
            current = self._layer.config.get(key, default)

            with Horizontal(classes="config-row"):
                yield Label(f"{label}:", classes="config-label")

                if config_type == "int":
                    yield Input(
                        value=str(current),
                        type="integer",
                        id=f"config-{key}",
                    )
                elif config_type == "float":
                    yield Input(
                        value=str(current),
                        type="number",
                        id=f"config-{key}",
                    )
                elif config_type == "bool":
                    yield Switch(value=bool(current), id=f"config-{key}")
                elif config_type == "select":
                    options = [(opt, opt) for opt in default]
                    yield Select(
                        options=options,
                        value=current if current in default else default[0],
                        id=f"config-{key}",
                    )

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id and event.input.id.startswith("config-"):
            key = event.input.id.replace("config-", "")
            try:
                if event.input.type == "integer":
                    value = int(event.value) if event.value else 0
                elif event.input.type == "number":
                    value = float(event.value) if event.value else 0.0
                else:
                    value = event.value
                self.post_message(self.ConfigChanged(self.index, key, value))
            except ValueError:
                pass

    @on(Switch.Changed)
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id and event.switch.id.startswith("config-"):
            key = event.switch.id.replace("config-", "")
            self.post_message(self.ConfigChanged(self.index, key, event.value))

    @on(Select.Changed)
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id and event.select.id.startswith("config-"):
            key = event.select.id.replace("config-", "")
            self.post_message(self.ConfigChanged(self.index, key, event.value))


class TopologyView(Static):
    """Visual representation of model topology."""

    DEFAULT_CSS = """
    TopologyView {
        width: 100%;
        height: 100%;
        padding: 1;
        background: #1E1E2E;
        border: round #45475A;
    }
    """

    def render_topology(self, layers: list[LayerConfig]) -> None:
        """Render the topology visualization."""
        if not layers:
            self.update("[dim]No layers defined[/]")
            return

        lines = ["[bold]Model Topology[/]", ""]

        for i, layer in enumerate(layers):
            icon = {
                "attention": "🔍",
                "mosaic": "🧩",
                "ffn": "⚡",
                "embedding": "📝",
                "norm": "📐",
                "output": "🎯",
            }.get(layer.layer_type, "📦")

            # Box drawing
            if i == 0:
                lines.append("    ┌─────────────────────┐")

            name = layer.name[:17] + "..." if len(layer.name) > 20 else layer.name.ljust(20)
            lines.append(f"    │ {icon} {name}│")

            if i < len(layers) - 1:
                lines.append("    ├─────────────────────┤")
                lines.append("    │         ↓           │")
                lines.append("    ├─────────────────────┤")
            else:
                lines.append("    └─────────────────────┘")

        self.update("\n".join(lines))


class ManifestBuilder(App[None]):
    """Manifest builder application."""

    CSS = """
    ManifestBuilder {
        background: #1E1E2E;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #left-panel {
        width: 35;
        background: #2D2D3D;
        border-right: solid #45475A;
    }

    #center-panel {
        width: 1fr;
        padding: 1;
    }

    #right-panel {
        width: 45;
        background: #2D2D3D;
        border-left: solid #45475A;
        padding: 1;
    }

    #layer-list {
        height: 1fr;
    }

    #add-layer-section {
        height: auto;
        padding: 1;
        border-top: solid #45475A;
    }

    #yaml-preview {
        width: 100%;
        height: 100%;
    }

    .section-title {
        text-style: bold;
        color: #A78BFA;
        padding: 1;
    }

    Button {
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "save", "Save"),
        Binding("n", "new_layer", "New Layer"),
        Binding("d", "delete_layer", "Delete Layer"),
        Binding("ctrl+up", "move_up", "Move Up"),
        Binding("ctrl+down", "move_down", "Move Down"),
    ]

    # State
    manifest: reactive[ManifestConfig] = reactive(ManifestConfig, recompose=False)
    selected_layer_index: reactive[int] = reactive(-1)

    def __init__(self, output_path: Path | str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_path = Path(output_path) if output_path else Path("manifest.yml")
        self.manifest = ManifestConfig()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # Left panel: Layer list
            with Vertical(id="left-panel"):
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
            with Vertical(id="center-panel"):
                with TabbedContent():
                    with TabPane("Configuration", id="config-tab"):
                        yield VerticalScroll(LayerConfigPanel(id="layer-config-panel"))

                    with TabPane("Topology", id="topology-tab"):
                        yield TopologyView(id="topology-view")

                    with TabPane("Training", id="training-tab"):
                        yield self._build_training_config()

            # Right panel: YAML preview
            with Vertical(id="right-panel"):
                yield Label("📄 YAML Preview", classes="section-title")
                yield TextArea(
                    self.manifest.to_yaml(),
                    language="yaml",
                    read_only=True,
                    id="yaml-preview",
                )
                yield Button("💾 Save Manifest", id="save-btn", variant="success")

        yield Footer()

    def _build_training_config(self) -> Vertical:
        """Build training configuration panel widget."""
        container = Vertical()
        # We need to add children after mount, so we'll use a VerticalScroll with static content
        # For now, return a simple container with the label
        return Vertical(
            Label("Training Configuration", classes="section-title"),
            Horizontal(
                Label("Batch Size:", classes="config-label"),
                Input(
                    value=str(self.manifest.batch_size),
                    type="integer",
                    id="train-batch-size",
                ),
                classes="config-row",
            ),
            Horizontal(
                Label("Block Size:", classes="config-label"),
                Input(
                    value=str(self.manifest.block_size),
                    type="integer",
                    id="train-block-size",
                ),
                classes="config-row",
            ),
            Horizontal(
                Label("Learning Rate:", classes="config-label"),
                Input(
                    value=str(self.manifest.lr),
                    type="number",
                    id="train-lr",
                ),
                classes="config-row",
            ),
            Horizontal(
                Label("Steps:", classes="config-label"),
                Input(
                    value=str(self.manifest.steps),
                    type="integer",
                    id="train-steps",
                ),
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
        """Initialize the builder."""
        self._refresh_layer_list()
        self._refresh_preview()

    def _refresh_layer_list(self) -> None:
        """Refresh the layer list view."""
        try:
            list_view = self.query_one("#layer-list", ListView)
            list_view.clear()
            for i, layer in enumerate(self.manifest.layers):
                list_view.append(LayerListItem(layer, i))
        except Exception:
            pass

    def _refresh_preview(self) -> None:
        """Refresh the YAML preview."""
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

    def _refresh_config_panel(self) -> None:
        """Refresh the layer config panel."""
        try:
            container = self.query_one("#config-tab VerticalScroll")
            old_panel = container.query_one(LayerConfigPanel)

            if 0 <= self.selected_layer_index < len(self.manifest.layers):
                layer = self.manifest.layers[self.selected_layer_index]
                new_panel = LayerConfigPanel(layer, self.selected_layer_index, id="layer-config-panel")
            else:
                new_panel = LayerConfigPanel(None, -1, id="layer-config-panel")

            old_panel.remove()
            container.mount(new_panel)
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

            # Create layer with default config
            defaults = {}
            for key, _, _, default in LAYER_CONFIGS.get(str(layer_type), []):
                if isinstance(default, list):
                    defaults[key] = default[0]
                else:
                    defaults[key] = default

            layer = LayerConfig(
                layer_type=str(layer_type),
                name=layer_name,
                config=defaults,
            )
            self.manifest.layers.append(layer)

            # Clear input
            self.query_one("#layer-name-input", Input).value = ""

            self._refresh_layer_list()
            self._refresh_preview()
        except Exception:
            pass

    @on(ListView.Selected)
    def on_layer_selected(self, event: ListView.Selected) -> None:
        """Handle layer selection."""
        if isinstance(event.item, LayerListItem):
            self.selected_layer_index = event.item.index
            self._refresh_config_panel()

    @on(LayerConfigPanel.ConfigChanged)
    def on_config_changed(self, event: LayerConfigPanel.ConfigChanged) -> None:
        """Handle layer config changes."""
        if 0 <= event.layer_index < len(self.manifest.layers):
            self.manifest.layers[event.layer_index].config[event.key] = event.value
            self._refresh_preview()

    @on(Button.Pressed, "#save-btn")
    def save_manifest(self) -> None:
        """Save the manifest to file."""
        self.output_path.write_text(self.manifest.to_yaml())
        self.notify(f"Saved to {self.output_path}", title="Manifest Saved")

    def action_quit(self) -> None:
        """Quit the builder."""
        self.exit()

    def action_save(self) -> None:
        """Save manifest."""
        self.save_manifest()

    def action_new_layer(self) -> None:
        """Focus new layer input."""
        try:
            self.query_one("#layer-name-input", Input).focus()
        except Exception:
            pass

    def action_delete_layer(self) -> None:
        """Delete selected layer."""
        if 0 <= self.selected_layer_index < len(self.manifest.layers):
            del self.manifest.layers[self.selected_layer_index]
            self.selected_layer_index = -1
            self._refresh_layer_list()
            self._refresh_config_panel()
            self._refresh_preview()

    def action_move_up(self) -> None:
        """Move selected layer up."""
        idx = self.selected_layer_index
        if idx > 0:
            self.manifest.layers[idx], self.manifest.layers[idx - 1] = (
                self.manifest.layers[idx - 1],
                self.manifest.layers[idx],
            )
            self.selected_layer_index = idx - 1
            self._refresh_layer_list()
            self._refresh_preview()

    def action_move_down(self) -> None:
        """Move selected layer down."""
        idx = self.selected_layer_index
        if 0 <= idx < len(self.manifest.layers) - 1:
            self.manifest.layers[idx], self.manifest.layers[idx + 1] = (
                self.manifest.layers[idx + 1],
                self.manifest.layers[idx],
            )
            self.selected_layer_index = idx + 1
            self._refresh_layer_list()
            self._refresh_preview()


def main(output: str | None = None) -> None:
    """Run the manifest builder."""
    app = ManifestBuilder(output_path=output)
    app.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manifest Builder")
    parser.add_argument("-o", "--output", type=str, default="manifest.yml", help="Output path")
    args = parser.parse_args()
    main(output=args.output)
