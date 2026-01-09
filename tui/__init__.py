"""Caramba TUI - Terminal user interface for AI agent interaction.

The unified TUI provides three views:
- Chat: Agent chat interface (Ctrl+1)
- Training: Real-time training metrics dashboard (Ctrl+2)
- Builder: Visual manifest/architecture builder (Ctrl+3)

Usage:
    caramba tui --url http://localhost:9000 --log runs/train.jsonl
"""

from caramba.tui.unified import CarambaApp, ChatView, TrainingView, BuilderView
from caramba.tui.unified import main
from caramba.tui.app import RootChatApp
from caramba.tui.sidebars import AgentStatus, ExpertStatus, AgentNode, AgentDetailModal, ToolDetailModal
from caramba.tui.training_dashboard import TrainingDashboard, TrainingMetrics
from caramba.tui.manifest_builder import ManifestBuilder, ManifestConfig, LayerConfig

__all__ = [
    # Unified TUI (primary entrypoint)
    "CarambaApp",
    "ChatView",
    "TrainingView",
    "BuilderView",
    "main",
    # Standalone apps
    "RootChatApp",
    "TrainingDashboard",
    "ManifestBuilder",
    # Shared components
    "AgentStatus",
    "ExpertStatus",
    "AgentNode",
    "AgentDetailModal",
    "ToolDetailModal",
    "TrainingMetrics",
    "ManifestConfig",
    "LayerConfig",
]
