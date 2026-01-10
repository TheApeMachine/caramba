"""Textual TUI for chatting with the Root agent via A2A.

This provides a modern chat-like interface with:
- Main chat viewport with styled message bubbles
- Sidebars showing agent hierarchy with health status
- Tool calls being made
- Slash command autocomplete
- Command palette (Ctrl+P)
- Keyboard shortcuts and command history
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    SendStreamingMessageRequest,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header
from textual.binding import Binding

from caramba.tui.styles import TUI_CSS
from caramba.tui.viewport import Viewport
from caramba.tui.input_bar import InputBar
from caramba.tui.sidebars import ExpertsSidebar, ToolsSidebar, StatusBar, ExpertStatus, AgentStatus, AgentDetailModal, ToolDetailModal, log_agent_event
from caramba.tui.command_palette import CommandPalette, HelpScreen
from caramba.tui.commands import Command

# Default polling interval for agent health checks (seconds)
HEALTH_CHECK_INTERVAL = 10.0


def get_base_url(url: str) -> str:
    """Extract the base URL (scheme + host + port) from a full URL.

    Examples:
        http://localhost:9000/sse -> http://localhost:9000
        http://localhost:9000 -> http://localhost:9000
    """
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


class RootChatApp(App):
    """Modern Textual TUI for chatting with Root agent."""

    CSS = TUI_CSS

    TITLE = "Caramba Chat"
    SUB_TITLE = "AI Agent Interface"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear", priority=True),
        Binding("f1", "help", "Help"),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "focus_next_panel", "Next Panel", show=False),
        Binding("shift+tab", "focus_previous_panel", "Prev Panel", show=False),
    ]

    def __init__(
        self,
        root_agent_url: str = "http://localhost:9000/sse",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.root_agent_url = root_agent_url
        self._viewport: Viewport | None = None
        self._experts_sidebar: ExpertsSidebar | None = None
        self._tools_sidebar: ToolsSidebar | None = None
        self._input_bar: InputBar | None = None
        self._status_bar: StatusBar | None = None
        self._is_streaming = False
        # Conversation context tracking for session persistence
        self._context_id: str = uuid4().hex  # Single conversation context
        self._task_id: str | None = None  # Current task ID from last response

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-sidebar"):
                yield ExpertsSidebar(id="experts-sidebar")
            with Vertical(id="chat-area"):
                yield Viewport(id="viewport")
                yield InputBar(id="input-bar")
            with Vertical(id="right-sidebar"):
                yield ToolsSidebar(id="tools-sidebar")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app components."""
        self._viewport = self.query_one("#viewport", Viewport)
        self._experts_sidebar = self.query_one("#experts-sidebar", ExpertsSidebar)
        self._tools_sidebar = self.query_one("#tools-sidebar", ToolsSidebar)
        self._input_bar = self.query_one("#input-bar", InputBar)
        self._status_bar = self.query_one("#status-bar", StatusBar)

        # Set initial status
        self._status_bar.set_agent_url(self.root_agent_url)

        # Focus the input
        self._input_bar.focus_input()

        # Initial agent hierarchy check
        self.refresh_agent_hierarchy()

        # Start periodic health checking
        self.set_interval(HEALTH_CHECK_INTERVAL, self._periodic_health_check)

    def _periodic_health_check(self) -> None:
        """Periodic callback for health checks."""
        self.refresh_agent_hierarchy()

    @work(exclusive=False)
    async def check_connection(self) -> None:
        """Check connection to the root agent."""
        if not self._status_bar:
            return

        base_url = get_base_url(self.root_agent_url)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try /health first, fall back to agent card
                response = await client.get(f"{base_url}/health")
                if response.status_code == 404:
                    response = await client.get(f"{base_url}/.well-known/agent-card.json")
                self._status_bar.set_connected(response.status_code == 200)
        except Exception:
            self._status_bar.set_connected(False)

    @work(exclusive=False)
    async def refresh_agent_hierarchy(self) -> None:
        """Fetch and display the agent hierarchy with health status."""
        if not self._experts_sidebar or not self._status_bar:
            return

        base_url = get_base_url(self.root_agent_url)
        root_error_msg = ""
        root_healthy = False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Indicate refresh without changing list layout.
                self._experts_sidebar.update_root_status(AgentStatus.CHECKING, "")

                # First, check root agent health
                # Try /health first, fall back to /.well-known/agent-card.json
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 404:
                        # Fallback to agent card endpoint (always exists on A2A servers)
                        response = await client.get(f"{base_url}/.well-known/agent-card.json")
                    root_healthy = response.status_code == 200
                    self._status_bar.set_connected(root_healthy)
                    if not root_healthy:
                        root_error_msg = f"HTTP {response.status_code}"
                except httpx.ConnectError:
                    root_error_msg = "connection refused"
                    self._status_bar.set_connected(False)
                except httpx.TimeoutException:
                    root_error_msg = "connection timeout"
                    self._status_bar.set_connected(False)
                except Exception as e:
                    root_error_msg = str(e)[:30] if str(e) else "unknown error"
                    self._status_bar.set_connected(False)

                # Try to get agent hierarchy from /agents/status endpoint
                # Use longer timeout since root agent pings all sub-agents
                sub_agents_data: dict[str, dict] = {}
                root_name = "Root"
                try:
                    response = await client.get(f"{base_url}/agents/status", timeout=15.0)
                    if response.status_code == 200:
                        data = response.json()
                        # Expected format: {"root": {...}, "sub_agents": {"name": {"healthy": bool, "error": "..."}, ...}}
                        root_info = data.get("root", {})
                        root_name = root_info.get("name", "Root")
                        teams_data = data.get("teams", {})
                        sub_agents_data = data.get("sub_agents", {})

                        # Prefer team-centric view when available.
                        if isinstance(teams_data, dict) and teams_data:
                            self._experts_sidebar.set_root_teams(
                                root_name,
                                AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                                teams_data,
                                root_error_msg,
                            )
                            # Populate agent health for all members (may appear multiple times).
                            for _, tinfo in teams_data.items():
                                agents = tinfo.get("agents", {}) if isinstance(tinfo, dict) else {}
                                if not isinstance(agents, dict):
                                    continue
                                for agent_name, info in agents.items():
                                    if not isinstance(info, dict):
                                        continue
                                    healthy = info.get("healthy", False)
                                    error = info.get("error", "")
                                    if error and len(error) > 30:
                                        error = error[:27] + "..."
                                    self._experts_sidebar.update_sub_agent_status(
                                        agent_name,
                                        AgentStatus.HEALTHY if healthy else AgentStatus.UNHEALTHY,
                                        error if not healthy else "",
                                    )
                            return

                        # Extract sub-agent names
                        sub_agent_names = list(sub_agents_data.keys())
                        # Build one more level: lead -> member list (if provided by server)
                        lead_children: dict[str, list[str]] = {}
                        for lead_name, info in sub_agents_data.items():
                            sub = info.get("sub_agents", {}) if isinstance(info, dict) else {}
                            if isinstance(sub, dict) and sub:
                                lead_children[lead_name] = list(sub.keys())

                        # Set root agent with sub-agents
                        self._experts_sidebar.set_root_agent(
                            root_name,
                            AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                            sub_agent_names,
                            lead_children if lead_children else None,
                            root_error_msg,
                        )

                        # Update each sub-agent status with error messages
                        for name, info in sub_agents_data.items():
                            healthy = info.get("healthy", False)
                            error = info.get("error", "")
                            # Shorten common error messages
                            if error:
                                if "Connection refused" in error or "ConnectError" in error:
                                    error = "connection refused"
                                elif "timeout" in error.lower():
                                    error = "connection timeout"
                                elif len(error) > 30:
                                    error = error[:27] + "..."
                            self._experts_sidebar.update_sub_agent_status(
                                name,
                                AgentStatus.HEALTHY if healthy else AgentStatus.UNHEALTHY,
                                error if not healthy else "",
                            )

                            # Update nested members if present.
                            nested = info.get("sub_agents", {}) if isinstance(info, dict) else {}
                            if isinstance(nested, dict) and nested:
                                for member_name, member_info in nested.items():
                                    if not isinstance(member_info, dict):
                                        continue
                                    m_ok = member_info.get("healthy", False)
                                    m_err = member_info.get("error", "")
                                    if m_err and len(m_err) > 30:
                                        m_err = m_err[:27] + "..."
                                    self._experts_sidebar.update_sub_agent_status(
                                        member_name,
                                        AgentStatus.HEALTHY if m_ok else AgentStatus.UNHEALTHY,
                                        m_err if not m_ok else "",
                                    )
                        return

                    elif response.status_code == 404:
                        # /agents/status not available, try to get name from agent card
                        try:
                            card_resp = await client.get(f"{base_url}/.well-known/agent-card.json")
                            if card_resp.status_code == 200:
                                card = card_resp.json()
                                root_name = card.get("name", "Root")
                        except Exception:
                            pass

                        # Show root without sub-agents (server needs rebuild for full feature)
                        self._experts_sidebar.set_root_agent(
                            root_name,
                            AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                            message=root_error_msg if root_error_msg else ("" if root_healthy else "rebuild server for hierarchy"),
                        )
                        return

                except Exception:
                    # Don't wipe the sidebar on transient failures; keep last known tree
                    # and just reflect root health.
                    self._experts_sidebar.update_root_status(
                        AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                        root_error_msg,
                    )
                    return

                # Fallback: just show root agent without hierarchy info
                self._experts_sidebar.set_root_agent(
                    root_name,
                    AgentStatus.HEALTHY if root_healthy else AgentStatus.UNHEALTHY,
                    message=root_error_msg,
                )

        except Exception as e:
            # Complete failure - mark everything as unhealthy
            error_msg = str(e)[:30] if str(e) else "unknown error"
            if self._experts_sidebar:
                self._experts_sidebar.set_root_agent(
                    "Root",
                    AgentStatus.UNHEALTHY,
                    message=error_msg,
                )

    @on(InputBar.MessageSubmitted, "#input-bar")
    def handle_message(self, event: InputBar.MessageSubmitted) -> None:
        """Handle user message submission."""
        if not self._viewport or self._is_streaming:
            return

        user_message = event.message.strip()
        if not user_message:
            return

        # Add user message to viewport
        self._viewport.add_user_message(user_message)

        # Show thinking indicator
        self._viewport.show_thinking()

        # Stream response from root agent
        self.stream_root_response(user_message)

    @on(InputBar.CommandInvoked, "#input-bar")
    def handle_command(self, event: InputBar.CommandInvoked) -> None:
        """Handle slash command invocation."""
        self.execute_command(event.command, event.args)

    def execute_command(self, command: Command, args: str = "") -> None:
        """Execute a slash command."""
        cmd_name = command.name

        if cmd_name == "help":
            self.action_help()
        elif cmd_name == "clear":
            self.action_clear_chat()
        elif cmd_name == "quit":
            self.action_quit()
        elif cmd_name == "status":
            self.refresh_agent_hierarchy()
            if self._viewport:
                self._viewport.add_system_message(
                    f"Refreshing agent status...\nRoot agent URL: {self.root_agent_url}"
                )
        elif cmd_name == "connect":
            if args:
                self.root_agent_url = args.strip()
                if self._status_bar:
                    self._status_bar.set_agent_url(self.root_agent_url)
                self.check_connection()
                if self._viewport:
                    self._viewport.add_system_message(
                        f"Connected to: {self.root_agent_url}"
                    )
        elif cmd_name == "export":
            if self._viewport:
                self._viewport.add_system_message(
                    "[dim]Export functionality coming soon...[/]"
                )
        elif cmd_name == "experts":
            if self._viewport:
                self._viewport.add_system_message(
                    "[dim]Expert listing coming soon...[/]"
                )
        elif cmd_name == "tools":
            if self._viewport:
                self._viewport.add_system_message(
                    "[dim]Tool listing coming soon...[/]"
                )
        elif cmd_name == "debug":
            if self._viewport:
                self._viewport.add_system_message(
                    "[dim]Debug mode toggled[/]"
                )
        else:
            if self._viewport:
                self._viewport.add_system_message(
                    f"[dim]Unknown command: /{cmd_name}[/]"
                )

    @work(exclusive=True)
    async def stream_root_response(self, user_message: str) -> None:
        """Stream response from root agent via A2A."""
        if not self._viewport:
            return

        self._is_streaming = True

        # Log the user message
        log_agent_event("Root", "message", f"User: {user_message[:200]}")

        # Show root agent as working
        if self._experts_sidebar:
            self._experts_sidebar.set_root_activity(ExpertStatus.RESPONDING)

        try:
            base_url = get_base_url(self.root_agent_url)
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as httpx_client:
                self._viewport.hide_thinking()
                self._viewport.start_streaming()

                card = await A2ACardResolver(httpx_client, base_url).get_agent_card()
                # Force the client to use the URL the user connected to, even if the
                # AgentCard.url is set to an internal container address.
                client = A2AClient(httpx_client=httpx_client, agent_card=card, url=base_url)

                payload: dict[str, Any] = {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": user_message}],
                        "messageId": uuid4().hex,
                        "contextId": self._context_id,  # Preserve conversation context
                    }
                }
                # Include task_id if we have one from a previous response
                if self._task_id:
                    payload["message"]["taskId"] = self._task_id

                request = SendStreamingMessageRequest(
                    id=uuid4().hex,
                    params=MessageSendParams(**payload),
                )

                async for chunk in client.send_message_streaming(request):
                    root = getattr(chunk, "root", None)
                    if not isinstance(root, SendStreamingMessageSuccessResponse):
                        continue

                    event = root.result

                    if isinstance(event, Message):
                        # Message-only response: render parts and finish.
                        for p in event.parts:
                            part_root = getattr(p, "root", None)
                            if getattr(part_root, "kind", None) == "text":
                                text = getattr(part_root, "text", "")
                                if text:
                                    self._viewport.stream_token(str(text))
                        continue

                    if isinstance(event, TaskStatusUpdateEvent):
                        # Capture task ID for conversation continuity
                        if hasattr(event, "task_id") and event.task_id:
                            self._task_id = event.task_id
                        if hasattr(event, "context_id") and event.context_id:
                            self._context_id = event.context_id

                        # Clear task_id if task reached terminal state so next message creates new task
                        state = getattr(event.status, "state", None)
                        if state in ("completed", "failed", "canceled"):
                            self._task_id = None

                        msg = getattr(event.status, "message", None)
                        parts = getattr(msg, "parts", None) if msg is not None else None
                        if parts:
                            for p in parts:
                                part_root = getattr(p, "root", None)
                                if getattr(part_root, "kind", None) == "text":
                                    text = getattr(part_root, "text", "")
                                    if text:
                                        # Check if this is a tool call notification
                                        if text.startswith("Calling tool:"):
                                            tool_name = text.replace("Calling tool:", "").strip()
                                            if self._tools_sidebar:
                                                self._tools_sidebar.add_tool_call(tool_name, "Root")
                                            log_agent_event("Root", "tool_call", f"Calling: {tool_name}")
                                            if self._experts_sidebar:
                                                self._experts_sidebar.set_root_activity(ExpertStatus.CONSULTING)
                                        else:
                                            # Regular text - show in viewport
                                            self._viewport.stream_token(str(text))
                                            log_agent_event("Root", "message", text[:200])
                                            if self._experts_sidebar:
                                                self._experts_sidebar.set_root_activity(ExpertStatus.RESPONDING)

                                            # Detect delegation to sub-agents
                                            self._detect_delegation(text)
                        continue

                    if isinstance(event, TaskArtifactUpdateEvent):
                        artifact = getattr(event, "artifact", None)
                        parts = getattr(artifact, "parts", None) if artifact is not None else None
                        if parts:
                            for p in parts:
                                part_root = getattr(p, "root", None)
                                if getattr(part_root, "kind", None) == "text":
                                    text = getattr(part_root, "text", "")
                                    if text:
                                        self._viewport.stream_token(str(text))
                        continue

                    if isinstance(event, Task):
                        # Capture task ID for conversation continuity
                        if event.id:
                            self._task_id = event.id
                        if event.context_id:
                            self._context_id = event.context_id

                        # Clear task_id if task reached terminal state so next message creates new task
                        status = getattr(event, "status", None)
                        state = getattr(status, "state", None) if status else None
                        if state in ("completed", "failed", "canceled"):
                            self._task_id = None
                            # Mark any pending tool calls as complete
                            if self._tools_sidebar:
                                self._tools_sidebar.mark_tool_complete(success=(state == "completed"))

                        # Optional snapshots; ignore unless they carry a status message.
                        msg = getattr(status, "message", None) if status is not None else None
                        parts = getattr(msg, "parts", None) if msg is not None else None
                        if parts:
                            for p in parts:
                                part_root = getattr(p, "root", None)
                                if getattr(part_root, "kind", None) == "text":
                                    text = getattr(part_root, "text", "")
                                    if text:
                                        # Check if this is a tool call notification
                                        if text.startswith("Calling tool:"):
                                            tool_name = text.replace("Calling tool:", "").strip()
                                            if self._tools_sidebar:
                                                self._tools_sidebar.add_tool_call(tool_name, "Root")
                                            log_agent_event("Root", "tool_call", f"Calling: {tool_name}")
                                            if self._experts_sidebar:
                                                self._experts_sidebar.set_root_activity(ExpertStatus.CONSULTING)
                                        else:
                                            self._viewport.stream_token(str(text))
                                            log_agent_event("Root", "message", text[:200])
                                            if self._experts_sidebar:
                                                self._experts_sidebar.set_root_activity(ExpertStatus.RESPONDING)

                                            # Detect delegation to sub-agents
                                            self._detect_delegation(text)
                        continue

                self._viewport.end_streaming()

                # Mark root agent as done
                if self._experts_sidebar:
                    self._experts_sidebar.set_root_activity(ExpertStatus.DONE)

        except httpx.ConnectError:
            self._viewport.hide_thinking()
            self._viewport.add_error_message(
                f"Could not connect to agent at {self.root_agent_url}\n\n"
                "Make sure the agent is running and try again."
            )
            if self._status_bar:
                self._status_bar.set_connected(False)
            if self._experts_sidebar:
                self._experts_sidebar.set_root_activity(ExpertStatus.ERROR)
        except httpx.HTTPStatusError as e:
            self._viewport.hide_thinking()
            # For streaming responses, we can't access .text directly
            try:
                error_body = e.response.text
            except Exception:
                error_body = "(response body not available)"
            self._viewport.add_error_message(
                f"HTTP Error: {e.response.status_code}\n{error_body}"
            )
            if self._experts_sidebar:
                self._experts_sidebar.set_root_activity(ExpertStatus.ERROR)
        except Exception as e:
            self._viewport.hide_thinking()
            self._viewport.add_error_message(f"Error: {str(e)}")
            if self._experts_sidebar:
                self._experts_sidebar.set_root_activity(ExpertStatus.ERROR)
        finally:
            self._is_streaming = False
            # Clear all activity indicators after a short delay
            self.set_timer(2.0, self._clear_activity_indicators)

    def _clear_activity_indicators(self) -> None:
        """Clear all activity indicators after request completes."""
        if self._experts_sidebar:
            self._experts_sidebar.clear_all_activity()

    def _detect_delegation(self, text: str) -> None:
        """Detect delegation messages and update sub-agent activity indicators."""
        if not self._experts_sidebar:
            return

        text_lower = text.lower()

        # Common delegation patterns
        delegation_patterns = [
            ("project manager", "project_manager"),
            ("project_manager", "project_manager"),
            ("product owner", "product_owner"),
            ("product_owner", "product_owner"),
            ("research lead", "research_lead"),
            ("research_lead", "research_lead"),
            ("architect", "architect"),
            ("developer", "developer"),
            ("writer", "writer"),
            ("reviewer", "reviewer"),
            ("tester", "tester"),
            ("ml expert", "ml_expert"),
            ("ml_expert", "ml_expert"),
            ("mathematician", "mathematician"),
            ("data scientist", "data_scientist"),
            ("data_scientist", "data_scientist"),
            ("catalyst", "catalyst"),
            ("researcher", "researcher"),
        ]

        # Check for delegation indicators
        is_delegation = any(phrase in text_lower for phrase in [
            "delegated", "requested", "asked", "assigned", "sent to",
            "i've delegated", "i've requested", "i've asked", "i've assigned",
            "delegating to", "asking", "requesting from",
        ])

        if is_delegation:
            for display_name, agent_name in delegation_patterns:
                if display_name in text_lower:
                    # Update the agent's activity indicator
                    self._experts_sidebar.update_agent_activity(agent_name, ExpertStatus.WAITING)
                    # Also log the delegation
                    log_agent_event(agent_name, "delegation", f"Received task from Root")
                    break

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_command_palette(self) -> None:
        """Open the command palette."""
        self.push_screen(CommandPalette(), self._handle_palette_result)

    def _handle_palette_result(self, result: Any) -> None:
        """Handle result from command palette."""
        # Focus input after palette closes
        if self._input_bar:
            self._input_bar.focus_input()

    @on(CommandPalette.CommandSelected)
    def handle_palette_command(self, event: CommandPalette.CommandSelected) -> None:
        """Handle command selected from palette."""
        self.execute_command(event.command)

    def action_help(self) -> None:
        """Show the help screen."""
        self.push_screen(HelpScreen())

    def action_clear_chat(self) -> None:
        """Clear the chat viewport and reset conversation context."""
        if self._viewport:
            self._viewport.clear_messages()
        if self._experts_sidebar:
            self._experts_sidebar.clear_all_activity()
        # Reset conversation context for a fresh start
        self._context_id = uuid4().hex
        self._task_id = None
        if self._tools_sidebar:
            self._tools_sidebar.clear_tools()

    def action_cancel(self) -> None:
        """Cancel current operation or close modal."""
        # Close any open modals
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

    def action_focus_next_panel(self) -> None:
        """Cycle focus between panels: input -> agents sidebar -> tools sidebar -> input."""
        focused = self.focused
        in_input = self._input_bar and focused and focused in self._input_bar.query("*")
        in_experts = self._experts_sidebar and focused and focused in self._experts_sidebar.query("*")
        in_tools = self._tools_sidebar and focused and focused in self._tools_sidebar.query("*")

        if in_input:
            # Move to agents sidebar
            if self._experts_sidebar:
                widgets = self._experts_sidebar._get_nav_widgets()
                if widgets:
                    widgets[0].focus()
                    return
            # Fallback to tools
            if self._tools_sidebar:
                tools = self._tools_sidebar._get_tool_widgets()
                if tools:
                    tools[0].focus()
                    return
        elif in_experts:
            # Move to tools sidebar
            if self._tools_sidebar:
                tools = self._tools_sidebar._get_tool_widgets()
                if tools:
                    tools[0].focus()
                    return
            # Fallback to input
            if self._input_bar:
                self._input_bar.focus_input()
        elif in_tools:
            # Move back to input
            if self._input_bar:
                self._input_bar.focus_input()
        else:
            # Default: focus input
            if self._input_bar:
                self._input_bar.focus_input()

    def action_focus_previous_panel(self) -> None:
        """Cycle focus backwards between panels."""
        focused = self.focused
        in_input = self._input_bar and focused and focused in self._input_bar.query("*")
        in_experts = self._experts_sidebar and focused and focused in self._experts_sidebar.query("*")
        in_tools = self._tools_sidebar and focused and focused in self._tools_sidebar.query("*")

        if in_input:
            # Move to tools sidebar
            if self._tools_sidebar:
                tools = self._tools_sidebar._get_tool_widgets()
                if tools:
                    tools[-1].focus()
                    return
            # Fallback to agents
            if self._experts_sidebar:
                widgets = self._experts_sidebar._get_nav_widgets()
                if widgets:
                    widgets[-1].focus()
                    return
        elif in_tools:
            # Move to agents sidebar
            if self._experts_sidebar:
                widgets = self._experts_sidebar._get_nav_widgets()
                if widgets:
                    widgets[-1].focus()
                    return
            # Fallback to input
            if self._input_bar:
                self._input_bar.focus_input()
        elif in_experts:
            # Move back to input
            if self._input_bar:
                self._input_bar.focus_input()
        else:
            if self._input_bar:
                self._input_bar.focus_input()

    @on(ExpertsSidebar.AgentInspectRequested)
    def handle_agent_inspect(self, event: ExpertsSidebar.AgentInspectRequested) -> None:
        """Handle request to inspect an agent."""
        self.show_agent_details(event.agent_name, event.is_root)

    @on(ToolsSidebar.ToolInspectRequested)
    def handle_tool_inspect(self, event: ToolsSidebar.ToolInspectRequested) -> None:
        """Handle request to inspect a tool call."""
        self.show_tool_details(event.tool_name, event.agent_name, event.args, event.result, event.status)

    def show_tool_details(
        self,
        tool_name: str,
        agent_name: str,
        args: dict[str, Any] | None,
        result: Any,
        status: str,
    ) -> None:
        """Show the tool detail modal."""
        # Remove any existing modals - use query() to get all instances
        for modal in self.query(ToolDetailModal):
            modal.remove()
        for modal in self.query(AgentDetailModal):
            modal.remove()

        # Create and mount new modal with unique ID based on timestamp
        from time import time
        modal = ToolDetailModal(
            tool_name,
            agent_name,
            args,
            result,
            status,
            id=f"tool-detail-modal-{int(time() * 1000)}",
        )
        self.mount(modal)
        modal.focus()

    def show_agent_details(self, agent_name: str, is_root: bool) -> None:
        """Show the agent detail modal."""
        # Remove any existing modals - use query() to get all instances
        for modal in self.query(AgentDetailModal):
            modal.remove()
        for modal in self.query(ToolDetailModal):
            modal.remove()

        # Create and mount new modal with unique ID based on timestamp
        from time import time
        modal_id = f"agent-detail-modal-{int(time() * 1000)}"
        modal = AgentDetailModal(agent_name, is_root, id=modal_id)
        self.mount(modal)
        modal.focus()

        # Fetch agent details
        self.fetch_agent_details(agent_name, is_root, modal_id)

    @work(exclusive=False)
    async def fetch_agent_details(self, agent_name: str, is_root: bool, modal_id: str) -> None:
        """Fetch detailed information about an agent from root."""
        try:
            modal = self.query_one(f"#{modal_id}", AgentDetailModal)
        except Exception:
            return  # Modal was closed

        base_url = get_base_url(self.root_agent_url)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/agents/details",
                    params={"name": agent_name},
                )
                if response.status_code == 200:
                    details = response.json()
                    modal.set_details(details)
                elif response.status_code == 404:
                    # Endpoint not available, show basic info from cached status
                    modal.set_details({
                        "healthy": True if is_root else False,
                        "url": base_url if is_root else f"http://{agent_name.lower().replace('_', '-')}:8001",
                        "error": "" if is_root else "Details endpoint not available (rebuild server)",
                        "activity": "unknown",
                    })
                else:
                    modal.set_error(f"HTTP {response.status_code}")
        except Exception as e:
            try:
                modal.set_error(str(e))
            except Exception:
                pass  # Modal was closed


def main() -> None:
    """Main entrypoint for the TUI."""
    import argparse

    parser = argparse.ArgumentParser(description="Caramba Chat TUI")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:9000",
        help="Root agent URL",
    )
    args = parser.parse_args()

    app = RootChatApp(root_agent_url=args.url)
    app.run()


if __name__ == "__main__":
    main()
