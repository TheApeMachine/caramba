"""Sidebar widgets for the Caramba TUI.

This module provides the left and right sidebars showing:
- Agent hierarchy with health status (left)
- Tool calls being made (right)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from enum import Enum

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding


class AgentStatus(Enum):
    """Health/connection status of an agent."""

    UNKNOWN = "unknown"      # Not yet checked
    HEALTHY = "healthy"      # Responding to health checks
    UNHEALTHY = "unhealthy"  # Not responding or error
    CHECKING = "checking"    # Currently checking status


class ExpertStatus(Enum):
    """Activity status of an expert agent during a task."""

    IDLE = "idle"
    CONSULTING = "consulting"
    RESPONDING = "responding"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentNode:
    """Represents an agent in the hierarchy with its status."""

    name: str
    status: AgentStatus = AgentStatus.UNKNOWN
    activity: ExpertStatus = ExpertStatus.IDLE
    children: list["AgentNode"] = field(default_factory=list)
    url: str = ""
    last_check: datetime | None = None


class AgentWidget(Static, can_focus=True):
    """Widget showing an agent in the hierarchy with health and activity status."""

    DEFAULT_CSS = """
    AgentWidget {
        height: auto;
        padding: 0 1;
        margin-bottom: 0;
    }

    AgentWidget.root-agent {
        margin-bottom: 1;
    }

    AgentWidget.sub-agent {
        padding-left: 3;
    }

    AgentWidget.healthy {
        color: #10B981;
    }

    AgentWidget.unhealthy {
        color: #EF4444;
    }

    AgentWidget.unknown {
        color: #6C7086;
    }

    AgentWidget.checking {
        color: #F59E0B;
    }

    AgentWidget.activity-consulting {
        background: rgba(245, 158, 11, 0.2);
    }

    AgentWidget.activity-responding {
        background: rgba(59, 130, 246, 0.2);
    }

    AgentWidget:focus {
        background: rgba(167, 139, 250, 0.3);
        text-style: bold;
    }

    AgentWidget.selected {
        background: rgba(167, 139, 250, 0.2);
    }
    """

    BINDINGS = [
        Binding("enter", "select_agent", "Inspect", show=False),
        Binding("space", "select_agent", "Inspect", show=False),
    ]

    class Selected(Message):
        """Message sent when an agent is selected for inspection."""

        def __init__(self, agent_name: str, is_root: bool) -> None:
            super().__init__()
            self.agent_name = agent_name
            self.is_root = is_root

    health_status: reactive[AgentStatus] = reactive(AgentStatus.UNKNOWN)
    activity_status: reactive[ExpertStatus] = reactive(ExpertStatus.IDLE)
    status_message: reactive[str] = reactive("")

    def __init__(
        self,
        agent_name: str,
        *,
        is_root: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.is_root = is_root
        self._last_update = datetime.now()

    def on_mount(self) -> None:
        if self.is_root:
            self.add_class("root-agent")
        else:
            self.add_class("sub-agent")
        self._update_display()

    def action_select_agent(self) -> None:
        """Handle agent selection."""
        self.post_message(self.Selected(self.agent_name, self.is_root))

    def on_click(self) -> None:
        """Handle click to select agent."""
        self.focus()
        self.post_message(self.Selected(self.agent_name, self.is_root))

    def watch_health_status(self, status: AgentStatus) -> None:
        """React to health status changes."""
        for s in AgentStatus:
            self.remove_class(s.value)
        self.add_class(status.value)
        self._update_display()

    def watch_activity_status(self, status: ExpertStatus) -> None:
        """React to activity status changes."""
        # Remove all activity classes
        for s in ExpertStatus:
            self.remove_class(f"activity-{s.value}")
        # Add current activity class if active
        if status in (ExpertStatus.CONSULTING, ExpertStatus.RESPONDING):
            self.add_class(f"activity-{status.value}")
        self._update_display()

    def watch_status_message(self, message: str) -> None:
        """React to status message changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the display based on current status."""
        # Health indicator
        health_icons = {
            AgentStatus.UNKNOWN: "[dim]?[/]",
            AgentStatus.HEALTHY: "[green]●[/]",
            AgentStatus.UNHEALTHY: "[red]○[/]",
            AgentStatus.CHECKING: "[yellow]◌[/]",
        }
        health_icon = health_icons.get(self.health_status, "[dim]?[/]")

        # Activity indicator (shown if active)
        activity_suffix = ""
        if self.activity_status == ExpertStatus.CONSULTING:
            activity_suffix = " [yellow]⟳[/]"
        elif self.activity_status == ExpertStatus.RESPONDING:
            activity_suffix = " [blue]◉[/]"
        elif self.activity_status == ExpertStatus.DONE:
            activity_suffix = " [green]✓[/]"
        elif self.activity_status == ExpertStatus.ERROR:
            activity_suffix = " [red]✗[/]"

        prefix = "" if self.is_root else "└─ "
        name_style = "bold" if self.is_root else ""
        name_display = f"[{name_style}]{self.agent_name}[/]" if name_style else self.agent_name

        # Build the display with optional status line
        lines = [f"{prefix}{health_icon} {name_display}{activity_suffix}"]

        if self.status_message:
            # Indent status message to align under the name
            indent = "   " if self.is_root else "      "
            lines.append(f"{indent}[dim italic]{self.status_message}[/]")

        self.update("\n".join(lines))

    def set_health(self, status: AgentStatus, message: str = "") -> None:
        """Set the agent's health status with optional message."""
        self.health_status = status
        # Auto-generate message if not provided
        if not message:
            if status == AgentStatus.CHECKING:
                message = "checking..."
            elif status == AgentStatus.UNHEALTHY:
                message = "not responding"
            elif status == AgentStatus.UNKNOWN:
                message = "waiting to connect"
            elif status == AgentStatus.HEALTHY:
                message = ""  # No message needed when healthy
        self.status_message = message
        self._last_update = datetime.now()

    def set_status_message(self, message: str) -> None:
        """Set a custom status message."""
        self.status_message = message

    def set_activity(self, status: str | ExpertStatus) -> None:
        """Set the agent's activity status."""
        if isinstance(status, str):
            try:
                status = ExpertStatus(status)
            except ValueError:
                status = ExpertStatus.IDLE
        self.activity_status = status
        self._last_update = datetime.now()


class ExpertsSidebar(Vertical, can_focus=True):
    """Left sidebar showing agent hierarchy with health status."""

    DEFAULT_CSS = """
    ExpertsSidebar {
        width: 100%;
        height: 100%;
    }

    ExpertsSidebar .sidebar-header {
        height: 3;
        background: #2D2D3D;
        padding: 1 2;
        text-style: bold;
        color: #A78BFA;
        border-bottom: solid #45475A;
    }

    ExpertsSidebar .agents-section {
        padding: 1;
        border-bottom: solid #45475A;
    }

    ExpertsSidebar .agents-section-header {
        color: #6C7086;
        margin-bottom: 1;
        text-style: italic;
    }

    ExpertsSidebar .agents-list {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }

    ExpertsSidebar .empty-state {
        color: #6C7086;
        text-align: center;
        padding: 2;
    }

    ExpertsSidebar:focus-within .sidebar-header {
        background: #3D3D4D;
    }
    """

    BINDINGS = [
        Binding("up", "focus_previous_agent", "Previous", show=False),
        Binding("down", "focus_next_agent", "Next", show=False),
        Binding("k", "focus_previous_agent", "Previous", show=False),
        Binding("j", "focus_next_agent", "Next", show=False),
    ]

    class AgentInspectRequested(Message):
        """Message sent when user wants to inspect an agent."""

        def __init__(self, agent_name: str, is_root: bool) -> None:
            super().__init__()
            self.agent_name = agent_name
            self.is_root = is_root

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._root_agent: AgentWidget | None = None
        self._sub_agents: dict[str, AgentWidget] = {}
        self._agent_order: list[str] = []  # Track order for navigation

    def compose(self) -> ComposeResult:
        yield Static("🤖 Agents", classes="sidebar-header")
        with VerticalScroll(classes="agents-list", id="agents-scroll"):
            yield Static("[dim]Connecting to agents...[/]", classes="empty-state", id="agents-empty")

    def set_root_agent(
        self,
        name: str,
        status: AgentStatus = AgentStatus.UNKNOWN,
        sub_agents: list[str] | None = None,
        message: str = "",
    ) -> None:
        """Set the root agent and its sub-agents."""
        agents_scroll = self.query_one("#agents-scroll", VerticalScroll)

        # Remove empty state if present
        try:
            empty = self.query_one("#agents-empty")
            empty.remove()
        except Exception:
            pass

        # Create or update root agent widget
        if self._root_agent is None:
            self._root_agent = AgentWidget(name, is_root=True, id="root-agent")
            agents_scroll.mount(self._root_agent)
        self._root_agent.set_health(status, message)

        # Build agent order for navigation (root first, then sub-agents)
        self._agent_order = [name]

        # Add sub-agents
        if sub_agents:
            for sub_name in sub_agents:
                self._agent_order.append(sub_name)
                if sub_name not in self._sub_agents:
                    widget = AgentWidget(sub_name, is_root=False, id=f"agent-{sub_name}")
                    self._sub_agents[sub_name] = widget
                    agents_scroll.mount(widget)

    def _get_agent_widgets(self) -> list[AgentWidget]:
        """Get all agent widgets in order."""
        widgets: list[AgentWidget] = []
        if self._root_agent:
            widgets.append(self._root_agent)
        for name in self._agent_order[1:]:  # Skip root (already added)
            if name in self._sub_agents:
                widgets.append(self._sub_agents[name])
        return widgets

    def action_focus_next_agent(self) -> None:
        """Focus the next agent in the list."""
        widgets = self._get_agent_widgets()
        if not widgets:
            return

        # Find currently focused widget
        focused = self.app.focused
        try:
            current_idx = widgets.index(focused)  # type: ignore
            next_idx = (current_idx + 1) % len(widgets)
        except (ValueError, TypeError):
            next_idx = 0

        widgets[next_idx].focus()

    def action_focus_previous_agent(self) -> None:
        """Focus the previous agent in the list."""
        widgets = self._get_agent_widgets()
        if not widgets:
            return

        # Find currently focused widget
        focused = self.app.focused
        try:
            current_idx = widgets.index(focused)  # type: ignore
            prev_idx = (current_idx - 1) % len(widgets)
        except (ValueError, TypeError):
            prev_idx = len(widgets) - 1

        widgets[prev_idx].focus()

    def on_agent_widget_selected(self, event: AgentWidget.Selected) -> None:
        """Handle agent selection and bubble up as sidebar event."""
        self.post_message(self.AgentInspectRequested(event.agent_name, event.is_root))

    def update_root_status(self, status: AgentStatus, message: str = "") -> None:
        """Update the root agent's health status."""
        if self._root_agent:
            self._root_agent.set_health(status, message)

    def update_sub_agent_status(self, name: str, status: AgentStatus, message: str = "") -> None:
        """Update a sub-agent's health status."""
        if name in self._sub_agents:
            self._sub_agents[name].set_health(status, message)

    def update_agent_activity(self, name: str, activity: str | ExpertStatus) -> None:
        """Update an agent's activity status (consulting, responding, etc.)."""
        # Check root agent first
        if self._root_agent and self._root_agent.agent_name == name:
            self._root_agent.set_activity(activity)
            return
        # Check sub-agents
        if name in self._sub_agents:
            self._sub_agents[name].set_activity(activity)

    def update_expert_status(self, name: str, status: ExpertStatus) -> None:
        """Backward-compatible alias for updating expert activity status."""
        self.update_agent_activity(name, status)

    def set_all_agents_checking(self) -> None:
        """Mark all agents as 'checking' during a status refresh."""
        if self._root_agent:
            self._root_agent.set_health(AgentStatus.CHECKING)
        for widget in self._sub_agents.values():
            widget.set_health(AgentStatus.CHECKING)

    def get_sub_agent_names(self) -> list[str]:
        """Get list of sub-agent names."""
        return list(self._sub_agents.keys())

    def set_root_activity(self, activity: ExpertStatus) -> None:
        """Set the root agent's activity status."""
        if self._root_agent:
            self._root_agent.set_activity(activity)

    def clear_all_activity(self) -> None:
        """Clear all activity indicators (reset to idle)."""
        if self._root_agent:
            self._root_agent.set_activity(ExpertStatus.IDLE)
        for widget in self._sub_agents.values():
            widget.set_activity(ExpertStatus.IDLE)


class ToolCallWidget(Static, can_focus=True):
    """Widget showing a single tool call."""

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: #2D2D3D;
        border: round #45475A;
    }

    ToolCallWidget .tool-header {
        margin-bottom: 0;
    }

    ToolCallWidget .tool-agent {
        color: #6C7086;
    }

    ToolCallWidget .tool-name {
        color: #3B82F6;
        text-style: bold;
    }

    ToolCallWidget .tool-time {
        color: #6C7086;
        text-align: right;
    }

    ToolCallWidget.pending {
        border: round #F59E0B;
    }

    ToolCallWidget.success {
        border: round #10B981;
    }

    ToolCallWidget.error {
        border: round #EF4444;
    }

    ToolCallWidget:focus {
        background: rgba(59, 130, 246, 0.3);
    }
    """

    BINDINGS = [
        Binding("enter", "select_tool", "Inspect", show=False),
        Binding("space", "select_tool", "Inspect", show=False),
    ]

    class Selected(Message):
        """Message sent when a tool call is selected for inspection."""

        def __init__(self, tool_name: str, agent_name: str, args: dict[str, Any] | None, result: Any, status: str) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.agent_name = agent_name
            self.args = args
            self.result = result
            self.status = status

    def __init__(
        self,
        tool_name: str,
        agent_name: str = "root",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.timestamp = datetime.now()
        self._status = "pending"
        self._args: dict[str, Any] | None = None
        self._result: Any = None
        self._error: str | None = None

    def on_mount(self) -> None:
        self._update_display()
        self.add_class("pending")

    def _update_display(self) -> None:
        """Update the display."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        status_icon = {
            "pending": "[yellow]◌[/]",
            "success": "[green]✓[/]",
            "error": "[red]✗[/]",
        }.get(self._status, "[dim]?[/]")
        self.update(
            f"{status_icon} [dim]{self.agent_name}[/] → [cyan bold]{self.tool_name}[/]\n"
            f"   [dim]{time_str}[/]"
        )

    def set_args(self, args: dict[str, Any]) -> None:
        """Set the tool call arguments."""
        self._args = args

    def set_result(self, success: bool = True, result: Any = None, error: str | None = None) -> None:
        """Mark the tool call as complete."""
        self.remove_class("pending")
        self._result = result
        self._error = error
        if success:
            self.add_class("success")
            self._status = "success"
        else:
            self.add_class("error")
            self._status = "error"
        self._update_display()

    def action_select_tool(self) -> None:
        """Handle tool selection."""
        self.post_message(self.Selected(
            self.tool_name,
            self.agent_name,
            self._args,
            self._result if self._status == "success" else self._error,
            self._status,
        ))

    def on_click(self) -> None:
        """Handle click to select tool."""
        self.focus()
        self.action_select_tool()


class ToolsSidebar(Vertical, can_focus=True):
    """Right sidebar showing tool calls."""

    DEFAULT_CSS = """
    ToolsSidebar {
        width: 100%;
        height: 100%;
    }

    ToolsSidebar .sidebar-header {
        height: 3;
        background: #2D2D3D;
        padding: 1 2;
        text-style: bold;
        color: #3B82F6;
        border-bottom: solid #45475A;
    }

    ToolsSidebar .tools-list {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }

    ToolsSidebar .empty-state {
        color: #6C7086;
        text-align: center;
        padding: 2;
    }

    ToolsSidebar:focus-within .sidebar-header {
        background: #3D3D4D;
    }
    """

    BINDINGS = [
        Binding("up", "focus_previous_tool", "Previous", show=False),
        Binding("down", "focus_next_tool", "Next", show=False),
        Binding("k", "focus_previous_tool", "Previous", show=False),
        Binding("j", "focus_next_tool", "Next", show=False),
    ]

    class ToolInspectRequested(Message):
        """Message sent when user wants to inspect a tool call."""

        def __init__(self, tool_name: str, agent_name: str, args: dict[str, Any] | None, result: Any, status: str) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.agent_name = agent_name
            self.args = args
            self.result = result
            self.status = status

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tools: list[ToolCallWidget] = []

    def compose(self) -> ComposeResult:
        yield Static("🔧 Tools", classes="sidebar-header")
        with VerticalScroll(classes="tools-list", id="tools-scroll"):
            yield Static("[dim]No tool calls yet[/]", classes="empty-state", id="tools-empty")

    def _get_tool_widgets(self) -> list[ToolCallWidget]:
        """Get all tool widgets."""
        return list(self._tools)

    def action_focus_next_tool(self) -> None:
        """Focus the next tool in the list."""
        widgets = self._get_tool_widgets()
        if not widgets:
            return

        focused = self.app.focused
        try:
            current_idx = widgets.index(focused)  # type: ignore
            next_idx = (current_idx + 1) % len(widgets)
        except (ValueError, TypeError):
            next_idx = 0

        widgets[next_idx].focus()

    def action_focus_previous_tool(self) -> None:
        """Focus the previous tool in the list."""
        widgets = self._get_tool_widgets()
        if not widgets:
            return

        focused = self.app.focused
        try:
            current_idx = widgets.index(focused)  # type: ignore
            prev_idx = (current_idx - 1) % len(widgets)
        except (ValueError, TypeError):
            prev_idx = len(widgets) - 1

        widgets[prev_idx].focus()

    def on_tool_call_widget_selected(self, event: ToolCallWidget.Selected) -> None:
        """Handle tool selection and bubble up as sidebar event."""
        self.post_message(self.ToolInspectRequested(
            event.tool_name,
            event.agent_name,
            event.args,
            event.result,
            event.status,
        ))

    def add_tool_call(self, tool_name: str, agent_name: str = "root", args: dict[str, Any] | None = None) -> ToolCallWidget:
        """Add a tool call to the sidebar."""
        # Remove empty state if present
        try:
            empty = self.query_one("#tools-empty")
            empty.remove()
        except Exception:
            pass

        tool = ToolCallWidget(tool_name, agent_name)
        if args:
            tool.set_args(args)
        self._tools.append(tool)

        tools_list = self.query_one("#tools-scroll")
        tools_list.mount(tool)
        tools_list.scroll_end(animate=False)

        return tool

    def mark_tool_complete(self, success: bool = True, result: Any = None, error: str | None = None) -> None:
        """Mark the most recent tool call as complete."""
        if self._tools:
            self._tools[-1].set_result(success, result, error)

    def clear_tools(self) -> None:
        """Clear all tool calls from the sidebar."""
        tools_list = self.query_one("#tools-scroll")
        for tool in list(self._tools):
            tool.remove()
        self._tools.clear()
        tools_list.mount(
            Static("[dim]No tool calls yet[/]", classes="empty-state", id="tools-empty")
        )


class StatusBar(Static):
    """Status bar showing connection status and other info."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: #1E1E2E;
        padding: 0 2;
        dock: bottom;
    }

    StatusBar .status-connected {
        color: #10B981;
    }

    StatusBar .status-disconnected {
        color: #EF4444;
    }

    StatusBar .status-connecting {
        color: #F59E0B;
    }
    """

    connected: reactive[bool] = reactive(False)
    agent_url: reactive[str] = reactive("http://localhost:9000")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self._update_display()

    def watch_connected(self, connected: bool) -> None:
        """React to connection status changes."""
        self._update_display()

    def watch_agent_url(self, url: str) -> None:
        """React to URL changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the status display."""
        if self.connected:
            status = "[green]●[/] Connected"
        else:
            status = "[red]○[/] Disconnected"

        self.update(f"{status} │ {self.agent_url}")

    def set_connected(self, connected: bool) -> None:
        """Set the connection status."""
        self.connected = connected

    def set_agent_url(self, url: str) -> None:
        """Set the agent URL."""
        self.agent_url = url


class AgentDetailModal(Vertical):
    """Modal panel showing detailed agent information."""

    DEFAULT_CSS = """
    AgentDetailModal {
        width: 80%;
        max-width: 100;
        height: 80%;
        max-height: 40;
        background: #1E1E2E;
        border: round #A78BFA;
        padding: 1 2;
        layer: modal;
        align: center middle;
    }

    AgentDetailModal .modal-header {
        height: 3;
        background: #2D2D3D;
        padding: 1 2;
        text-style: bold;
        color: #A78BFA;
        margin-bottom: 1;
    }

    AgentDetailModal .modal-content {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
        background: #2D2D3D;
        border: round #45475A;
    }

    AgentDetailModal .modal-footer {
        height: 3;
        padding: 1;
        text-align: center;
        color: #6C7086;
    }

    AgentDetailModal .detail-section {
        margin-bottom: 1;
    }

    AgentDetailModal .detail-label {
        color: #A78BFA;
        text-style: bold;
    }

    AgentDetailModal .detail-value {
        color: #CDD6F4;
    }

    AgentDetailModal .error-text {
        color: #EF4444;
    }

    AgentDetailModal .success-text {
        color: #10B981;
    }

    AgentDetailModal .loading-text {
        color: #F59E0B;
        text-style: italic;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("q", "close", "Close", show=False),
    ]

    def __init__(
        self,
        agent_name: str,
        is_root: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.is_root = is_root
        self._content: Static | None = None

    def compose(self) -> ComposeResult:
        icon = "🤖" if self.is_root else "👤"
        yield Static(f"{icon} {self.agent_name}", classes="modal-header")
        with VerticalScroll(classes="modal-content"):
            yield Static("[yellow italic]Loading agent details...[/]", classes="loading-text", id="agent-detail-content")
        yield Static("[dim]Press [bold]Esc[/] to close[/]", classes="modal-footer")

    def on_mount(self) -> None:
        self._content = self.query_one("#agent-detail-content", Static)

    def set_loading(self) -> None:
        """Show loading state."""
        if self._content:
            self._content.update("[yellow italic]Loading agent details...[/]")

    def set_details(self, details: dict[str, Any]) -> None:
        """Set the agent details to display."""
        if not self._content:
            return

        lines: list[str] = []

        # Status
        healthy = details.get("healthy", False)
        status_icon = "[green]●[/]" if healthy else "[red]○[/]"
        status_text = "Healthy" if healthy else "Unhealthy"
        lines.append(f"[bold #A78BFA]Status:[/] {status_icon} {status_text}")
        lines.append("")

        # URL
        url = details.get("url", "N/A")
        lines.append(f"[bold #A78BFA]URL:[/] {url}")
        lines.append("")

        # Error (if any)
        error = details.get("error", "")
        if error:
            lines.append("[bold #A78BFA]Error:[/]")
            lines.append(f"[red]{error}[/]")
            lines.append("")

        # Last response (if any)
        last_response = details.get("last_response", "")
        if last_response:
            lines.append("[bold #A78BFA]Last Response:[/]")
            # Truncate long responses
            if len(last_response) > 500:
                last_response = last_response[:500] + "..."
            lines.append(f"[dim]{last_response}[/]")
            lines.append("")

        # Activity
        activity = details.get("activity", "idle")
        lines.append(f"[bold #A78BFA]Activity:[/] {activity}")

        # Type
        agent_type = details.get("type", "")
        if agent_type:
            lines.append(f"[bold #A78BFA]Type:[/] {agent_type}")

        self._content.update("\n".join(lines))

    def set_error(self, error: str) -> None:
        """Show error state."""
        if self._content:
            self._content.update(f"[red]Error loading agent details:[/]\n\n{error}")

    def action_close(self) -> None:
        """Close the modal."""
        self.remove()


class ToolDetailModal(Vertical):
    """Modal panel showing detailed tool call information."""

    DEFAULT_CSS = """
    ToolDetailModal {
        width: 80%;
        max-width: 120;
        height: 80%;
        max-height: 50;
        background: #1E1E2E;
        border: round #3B82F6;
        padding: 1 2;
        layer: modal;
        align: center middle;
    }

    ToolDetailModal .modal-header {
        height: 3;
        background: #2D2D3D;
        padding: 1 2;
        text-style: bold;
        color: #3B82F6;
        margin-bottom: 1;
    }

    ToolDetailModal .modal-content {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
        background: #2D2D3D;
        border: round #45475A;
    }

    ToolDetailModal .modal-footer {
        height: 3;
        padding: 1;
        text-align: center;
        color: #6C7086;
    }

    ToolDetailModal .section-header {
        color: #3B82F6;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }

    ToolDetailModal .code-block {
        background: #1E1E2E;
        padding: 1;
        margin: 0 0 1 0;
        color: #CDD6F4;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("q", "close", "Close", show=False),
    ]

    def __init__(
        self,
        tool_name: str,
        agent_name: str,
        args: dict[str, Any] | None = None,
        result: Any = None,
        status: str = "pending",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.args = args
        self.result = result
        self.status = status
        self._content: Static | None = None

    def compose(self) -> ComposeResult:
        status_icon = {
            "pending": "◌",
            "success": "✓",
            "error": "✗",
        }.get(self.status, "?")
        yield Static(f"🔧 {self.tool_name} {status_icon}", classes="modal-header")
        with VerticalScroll(classes="modal-content"):
            yield Static("", id="tool-detail-content")
        yield Static("[dim]Press [bold]Esc[/] to close[/]", classes="modal-footer")

    def on_mount(self) -> None:
        self._content = self.query_one("#tool-detail-content", Static)
        self._update_content()

    def _format_json(self, obj: Any, indent: int = 2) -> str:
        """Format an object as pretty JSON."""
        import json
        try:
            return json.dumps(obj, indent=indent, default=str)
        except Exception:
            return str(obj)

    def _update_content(self) -> None:
        """Update the modal content."""
        if not self._content:
            return

        lines: list[str] = []

        # Status
        status_color = {
            "pending": "yellow",
            "success": "green",
            "error": "red",
        }.get(self.status, "dim")
        lines.append(f"[bold #3B82F6]Status:[/] [{status_color}]{self.status}[/]")
        lines.append(f"[bold #3B82F6]Agent:[/] {self.agent_name}")
        lines.append("")

        # Arguments
        lines.append("[bold #3B82F6]Arguments:[/]")
        if self.args:
            args_str = self._format_json(self.args)
            # Truncate if too long
            if len(args_str) > 2000:
                args_str = args_str[:2000] + "\n... (truncated)"
            lines.append(f"[dim]{args_str}[/]")
        else:
            lines.append("[dim](no arguments)[/]")
        lines.append("")

        # Result/Error
        if self.status == "success":
            lines.append("[bold #3B82F6]Result:[/]")
            if self.result is not None:
                result_str = self._format_json(self.result)
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "\n... (truncated)"
                lines.append(f"[green]{result_str}[/]")
            else:
                lines.append("[dim](no result)[/]")
        elif self.status == "error":
            lines.append("[bold #3B82F6]Error:[/]")
            lines.append(f"[red]{self.result or '(unknown error)'}[/]")
        elif self.status == "pending":
            lines.append("[yellow italic]Tool call in progress...[/]")
            lines.append("")
            lines.append("[dim]The tool is currently executing. Check back later for results.[/]")

        self._content.update("\n".join(lines))

    def action_close(self) -> None:
        """Close the modal."""
        self.remove()
