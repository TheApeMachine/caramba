"""Remote agent connections for A2A protocol.

Manages connections to remote agents, enabling the root agent to
delegate tasks to team leads and for leads to delegate to members.

Supports both synchronous (blocking) and asynchronous (fire-and-forget with
push notifications) communication patterns.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, A2AClient, Client, ClientConfig, ClientFactory

# Timeout configuration for agent-to-agent communication
# LLM streaming responses can take a long time, so we need generous timeouts
A2A_TIMEOUT = httpx.Timeout(
    connect=10.0,      # Connection timeout
    read=300.0,        # Read timeout (5 minutes for long LLM responses)
    write=30.0,        # Write timeout
    pool=10.0,         # Pool timeout
)
from a2a.types import (
    AgentCard,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    Role,
    SendStreamingMessageRequest,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
    TransportProtocol,
)

from .types import AgentHealth, AgentState

# Type alias for task update callbacks
TaskCallback = Callable[[Task], Coroutine[Any, Any, None]]


@dataclass
class PendingTask:
    """Tracks a pending async task with its callback."""
    task_id: str
    agent_name: str
    context_id: str | None
    callback: TaskCallback | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


class PendingTaskManager:
    """Manages pending async tasks and their callbacks.
    
    When a task update is received via push notification, the manager
    looks up the callback and invokes it.
    """
    
    def __init__(self) -> None:
        self._pending: dict[str, PendingTask] = {}
        self._lock = asyncio.Lock()
    
    async def register(
        self,
        task_id: str,
        agent_name: str,
        context_id: str | None = None,
        callback: TaskCallback | None = None,
    ) -> None:
        """Register a pending task."""
        async with self._lock:
            self._pending[task_id] = PendingTask(
                task_id=task_id,
                agent_name=agent_name,
                context_id=context_id,
                callback=callback,
            )
    
    async def handle_update(self, task: Task) -> bool:
        """Handle a task update from a push notification.
        
        Returns True if the task was found and handled.
        """
        async with self._lock:
            pending = self._pending.get(task.id)
            if not pending:
                return False
            
            # If terminal state, remove from pending
            if task.status.state in (
                TaskState.completed,
                TaskState.failed,
                TaskState.canceled,
            ):
                del self._pending[task.id]
        
        # Invoke callback outside lock
        if pending.callback:
            try:
                _logger.info(f"TRACING [Callback] Invoking callback for task {task.id}")
                await pending.callback(task)
                _logger.info(f"TRACING [Callback] Callback completed for task {task.id}")
            except Exception as e:
                _logger.error(f"TRACING [Callback] Error in task callback for {task.id}: {e}", exc_info=True)
        else:
            _logger.warning(f"TRACING [Callback] No callback registered for task {task.id}")
        
        return True
    
    def get_pending(self, task_id: str) -> PendingTask | None:
        """Get a pending task by ID."""
        return self._pending.get(task_id)
    
    def list_pending(self) -> list[PendingTask]:
        """List all pending tasks."""
        return list(self._pending.values())


# Global pending task manager
_pending_tasks = PendingTaskManager()

# Set up logging
_logger = logging.getLogger("caramba.connection")


def get_pending_task_manager() -> PendingTaskManager:
    """Get the global pending task manager."""
    return _pending_tasks


async def handle_task_notification(task: Task) -> bool:
    """Handle an incoming task notification from a push webhook.
    
    This should be called by the webhook endpoint when a task update
    is received.
    
    Args:
        task: The task update from the notification.
        
    Returns:
        True if the task was found and handled.
    """
    return await _pending_tasks.handle_update(task)


class RemoteAgent:
    """Connection to a single remote agent.

    Handles A2A communication with a remote agent including
    message sending, task management, and health checking.
    """

    def __init__(
        self,
        card: AgentCard,
        client: Client,
    ) -> None:
        """Initialize a remote agent connection.

        Args:
            card: The agent's A2A card.
            client: The A2A client for this agent.
        """
        self.card = card
        self.client = client
        self._pending_tasks: set[str] = set()

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.card.name

    @property
    def description(self) -> str:
        """Get the agent description."""
        return self.card.description

    @property
    def url(self) -> str:
        """Get the agent URL."""
        return self.card.url or ""

    async def send_message(
        self,
        text: str,
        context_id: str | None = None,
        task_id: str | None = None,
        message_id: str | None = None,
    ) -> Task | Message | None:
        """Send a message to this agent.

        Args:
            text: The message text.
            context_id: Optional context ID for conversation threading.
            task_id: Optional task ID to continue.
            message_id: Optional message ID.

        Returns:
            The resulting Task or Message, or None on error.
        """
        _logger.info(f"Sending message to {self.name} at {self.url}")
        _logger.debug(f"Message: {text[:200]}...")
        
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=message_id or str(uuid.uuid4()),
            context_id=context_id,
            task_id=task_id,
        )

        last_task: Task | None = None
        last_error: str = ""
        event_count = 0
        try:
            async for event in self.client.send_message(message):
                event_count += 1
                _logger.debug(f"Received event {event_count}: type={type(event).__name__}")
                if isinstance(event, Message):
                    _logger.info(f"Got Message response from {self.name}")
                    return event
                if isinstance(event, tuple) and len(event) > 0:
                    task = event[0]
                    if isinstance(task, Task):
                        _logger.debug(f"Got Task: id={task.id}, state={task.status.state}")
                        if self._is_terminal(task):
                            _logger.info(f"Got terminal Task from {self.name}: state={task.status.state}")
                            return task
                        last_task = task
        except Exception as e:
            last_error = str(e)
            _logger.error(f"Error sending message to {self.name}: {e}", exc_info=True)
            # Return a failed task with error details instead of None
            return Task(
                id=message.task_id or str(uuid.uuid4()),
                context_id=message.context_id or str(uuid.uuid4()),
                status=TaskStatus(
                    state=TaskState.failed,
                    message=Message(
                        role=Role.agent,
                        parts=[Part(root=TextPart(text=f"Communication error: {last_error}"))],
                        message_id=str(uuid.uuid4()),
                    ),
                ),
            )

        _logger.info(f"Message exchange complete. Events received: {event_count}, last_task: {last_task is not None}")
        return last_task

    def _is_terminal(self, task: Task) -> bool:
        """Check if a task is in a terminal state."""
        return task.status.state in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.input_required,
            TaskState.unknown,
        ]

    async def send_message_async(
        self,
        text: str,
        webhook_url: str,
        context_id: str | None = None,
        task_id: str | None = None,
        message_id: str | None = None,
        callback: TaskCallback | None = None,
    ) -> str:
        """Send a message asynchronously (fire-and-forget).
        
        The task will be submitted and a task ID returned immediately.
        When the task updates, a push notification will be sent to the
        webhook_url. The callback will be invoked when updates are received.
        
        Args:
            text: The message text.
            webhook_url: URL to receive push notifications.
            context_id: Optional context ID for conversation threading.
            task_id: Optional task ID to continue.
            message_id: Optional message ID.
            callback: Optional callback for task updates.
            
        Returns:
            The task ID for tracking.
        """
        _logger.info(f"TRACING [Generic] Sending async message to agent='{self.name}' url='{self.url}'")
        _logger.info(f"TRACING [Generic] Webhook params: url='{webhook_url}'")
        _logger.debug(f"TRACING [Generic] Message preview: {text[:200]}...")
        
        msg_id = message_id or str(uuid.uuid4())
        new_task_id = task_id or str(uuid.uuid4())
        ctx_id = context_id or str(uuid.uuid4())
        
        # Task ID for the message: only set if explicitly provided (continuation)
        # If None, server will create a new task
        msg_task_id = task_id if task_id else None
        
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=msg_id,
            context_id=ctx_id,
            task_id=msg_task_id,
        )
        
        # Configure for non-blocking with push notifications
        config = MessageSendConfiguration(
            blocking=False,
            push_notification_config=PushNotificationConfig(
                url=webhook_url,
                token=new_task_id,  # Use task ID as token for verification
            ),
        )
        
        # Register the pending task before sending
        await _pending_tasks.register(
            task_id=new_task_id,
            agent_name=self.name,
            context_id=ctx_id,
            callback=callback,
        )
        _logger.info(f"TRACING [Generic] Registered pending task: id='{new_task_id}' agent='{self.name}' has_callback={callback is not None}")
        
        try:
            # Send non-blocking - just get initial task acknowledgment
            _logger.info(f"TRACING [Generic] Initiating client.send_message stream...")
            async for event in self.client.send_message(
                message,
                request_metadata={"configuration": config.model_dump(exclude_none=True)},
            ):
                if isinstance(event, tuple) and len(event) > 0:
                    task = event[0]
                    if isinstance(task, Task):
                        _logger.info(f"Async task submitted: {task.id} state={task.status.state}")
                        # Update task ID if server assigned a different one
                        if task.id != new_task_id:
                            await _pending_tasks.register(
                                task_id=task.id,
                                agent_name=self.name,
                                context_id=task.context_id,
                                callback=callback,
                            )
                            self._pending_tasks.add(task.id)
                            _logger.info(f"TRACING [Generic] Task ID updated from {new_task_id} to {task.id}")
                            new_task_id = task.id
                        # Don't wait for terminal state - return immediately
                        if task.status.state == TaskState.submitted:
                            _logger.info(f"TRACING [Generic] Task submitted successfully: {task.id}")
                            break
                        # If already working or completed, still return
                        _logger.info(f"TRACING [Generic] Task state is {task.status.state}, returning early.")
                        break
        except Exception as e:
            _logger.error(f"TRACING [Generic] Error submitting async task to {self.name}: {e}", exc_info=True)
            # Still return task ID so caller can track
        
        return new_task_id

    async def check_health(self) -> AgentHealth:
        """Check the health of this remote agent.

        Returns:
            AgentHealth with current status.
        """
        try:
            # Try to ping the agent's health endpoint
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = self.url.rstrip("/")
                response = await client.get(f"{url}/health")
                if response.status_code == 404:
                    response = await client.get(f"{url}/.well-known/agent-card.json")

                return AgentHealth(
                    name=self.name,
                    healthy=response.status_code == 200,
                    url=self.url,
                    activity=AgentState.IDLE,
                )
        except httpx.ConnectError:
            return AgentHealth(
                name=self.name,
                healthy=False,
                error="connection refused",
                url=self.url,
            )
        except httpx.TimeoutException:
            return AgentHealth(
                name=self.name,
                healthy=False,
                error="connection timeout",
                url=self.url,
            )
        except Exception as e:
            return AgentHealth(
                name=self.name,
                healthy=False,
                error=str(e)[:30],
                url=self.url,
            )


class ConnectionManager:
    """Manages connections to multiple remote agents.

    Provides a centralized way to connect to, communicate with,
    and monitor remote agents in the system.
    """

    def __init__(self, httpx_client: httpx.AsyncClient | None = None) -> None:
        """Initialize the connection manager.

        Args:
            httpx_client: Optional httpx client for reuse.
        """
        self._httpx_client = httpx_client
        self._owns_client = httpx_client is None
        self._connections: dict[str, RemoteAgent] = {}
        self._client_factory: ClientFactory | None = None

    async def __aenter__(self) -> ConnectionManager:
        """Async context manager entry."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=A2A_TIMEOUT)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._owns_client and self._httpx_client:
            await self._httpx_client.aclose()

    def _get_factory(self) -> ClientFactory:
        """Get or create the client factory."""
        if self._client_factory is None:
            if self._httpx_client is None:
                raise RuntimeError("ConnectionManager must be used as async context")
            config = ClientConfig(
                httpx_client=self._httpx_client,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                ],
            )
            self._client_factory = ClientFactory(config)
        return self._client_factory

    async def connect(self, url: str) -> RemoteAgent:
        """Connect to a remote agent by URL.

        Args:
            url: The agent's base URL.

        Returns:
            The RemoteAgent connection.

        Raises:
            Exception: If connection fails.
        """
        _logger.info(f"Connecting to agent at {url}")
        
        if self._httpx_client is None:
            _logger.error("ConnectionManager._httpx_client is None!")
            raise RuntimeError("ConnectionManager must be used as async context")

        # Check if already connected
        for agent in self._connections.values():
            if agent.url == url:
                _logger.debug(f"Already connected to {url} as {agent.name}")
                return agent

        # Resolve agent card
        _logger.debug(f"Resolving agent card from {url}")
        resolver = A2ACardResolver(self._httpx_client, url)
        card = await resolver.get_agent_card()
        _logger.info(f"Got agent card: name={card.name}, transport={card.preferred_transport}")

        # Create client
        factory = self._get_factory()
        client = factory.create(card)

        # Create connection
        agent = RemoteAgent(card, client)
        self._connections[card.name] = agent
        return agent

    async def connect_by_card(self, card: AgentCard) -> RemoteAgent:
        """Connect to a remote agent using its card.

        Args:
            card: The agent's A2A card.

        Returns:
            The RemoteAgent connection.
        """
        if card.name in self._connections:
            return self._connections[card.name]

        factory = self._get_factory()
        client = factory.create(card)
        agent = RemoteAgent(card, client)
        self._connections[card.name] = agent
        return agent

    def get(self, name: str) -> RemoteAgent | None:
        """Get a connection by agent name.

        Args:
            name: The agent name.

        Returns:
            The RemoteAgent or None if not connected.
        """
        return self._connections.get(name)

    def list_agents(self) -> list[dict[str, str]]:
        """List all connected agents.

        Returns:
            List of agent info dictionaries.
        """
        return [
            {"name": agent.name, "description": agent.description}
            for agent in self._connections.values()
        ]

    async def check_all_health(self) -> dict[str, AgentHealth]:
        """Check health of all connected agents.

        Returns:
            Dictionary mapping agent name to health status.
        """
        tasks = [agent.check_health() for agent in self._connections.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health: dict[str, AgentHealth] = {}
        for result in results:
            if isinstance(result, AgentHealth):
                health[result.name] = result
            elif isinstance(result, Exception):
                # Handle exceptions from failed health checks
                pass

        return health
