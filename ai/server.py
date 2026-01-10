"""A2A Server implementation.

Provides the HTTP server for exposing agents via the A2A protocol,
including streaming, health checks, and hierarchy status.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import litellm
import uvicorn
import os  # Added import

# Timeout configuration for agent-to-agent communication
# LLM streaming responses can take a long time, so we need generous timeouts
A2A_TIMEOUT = httpx.Timeout(
    connect=10.0,      # Connection timeout
    read=300.0,        # Read timeout (5 minutes for long LLM responses)
    write=30.0,        # Write timeout
    pool=10.0,         # Pool timeout
)

# LiteLLM timeout for LLM API calls (even longer for complex reasoning)
LITELLM_TIMEOUT = httpx.Timeout(
    connect=30.0,      # Connection timeout
    read=600.0,        # Read timeout (10 minutes for long reasoning chains)
    write=60.0,        # Write timeout
    pool=30.0,         # Pool timeout
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskStore
from a2a.types import AgentCard
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import BaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from caramba.ai.agent import Agent
from caramba.ai.connection import get_pending_task_manager, handle_task_notification
from caramba.ai.executor import StreamingExecutor
from caramba.ai.lead import LeadAgent
from caramba.ai.push_notifications import (
    HttpPushNotificationSender,
    InMemoryPushNotificationConfigStore,
)
from caramba.ai.root import RootAgent
from caramba.ai.session_store import get_shared_session_service
from caramba.ai.task_store import get_shared_task_store

_logger = logging.getLogger(__name__)


class CarambaAgentExecutor(StreamingExecutor):
    """Agent executor with streaming support for Caramba agents.

    Extends StreamingExecutor to provide real-time updates during
    task execution.
    """

    pass


class AgentServer:
    """A2A-compatible server for an agent.

    Wraps an Agent, LeadAgent, or RootAgent and exposes it via HTTP with
    streaming support, health checks, and status endpoints.
    """

    def __init__(
        self,
        agent: Agent | LeadAgent | RootAgent,
        host: str = "0.0.0.0",
        port: int = 9000,
        task_store: TaskStore | None = None,
        session_service: BaseSessionService | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            agent: The agent to serve.
            host: Host to bind to.
            port: Port to bind to.
            task_store: Shared task store for A2A task handoff.
            session_service: Shared session service for conversation persistence.
            httpx_client: Shared httpx client for push notifications.
        """
        self.agent = agent
        self.host = host
        self.port = port
        self.task_store = task_store
        self.httpx_client = httpx_client

        # ADK services - use provided session service or fall back to in-memory
        self.session_service = session_service or InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        self.memory_service = InMemoryMemoryService()

        # Push notification support for A2A async operations
        self.push_config_store = InMemoryPushNotificationConfigStore()
        self.push_sender = HttpPushNotificationSender(
            self.push_config_store,
            http_client=httpx_client,
        )

        # Build the agent based on type
        if isinstance(agent, RootAgent):
            self.adk_agent = agent.create_agent()
            self.agent_name = agent.config.name
        elif isinstance(agent, LeadAgent):
            self.adk_agent = agent.create_agent()
            self.agent_name = agent.config.name
        else:
            self.adk_agent = agent.build()
            self.agent_name = agent.name

        # Create runner
        self.runner = Runner(
            agent=self.adk_agent,
            app_name="caramba",
            session_service=self.session_service,
            artifact_service=self.artifact_service,
            memory_service=self.memory_service,
        )

        # Create executor
        self.executor = CarambaAgentExecutor(self.runner, self.agent_name)

        # Build the app
        self._app: Starlette | None = None

        # Background worker task
        self._worker_task: asyncio.Task | None = None

    async def _run_background_worker(self) -> None:
        """Run background worker to process pending tasks from the queue.
        
        This acts as the consumer for async A2A tasks.
        """
        _logger.info(f"TRACING [Worker] Starting background worker for {self.agent_name}")
        
        # We need to access the underlying TaskQueue if using DatabaseTaskStore
        from a2a.server.tasks import DatabaseTaskStore
        from caramba.core.task_queue import TaskQueue

        queue: TaskQueue | None = None
        
        # If we have a DB task store, we can use its engine to create a TaskQueue helper
        if isinstance(self.task_store, DatabaseTaskStore):
            # Hack: we reconstruct TaskQueue from engine URL if possible, 
            # or we rely on TaskQueue having been initialized as a singleton.
            # But core.task_queue.TaskQueue takes a DSN string.
            # Let's try to get the DSN from environment or connection.
            from caramba.ai.task_store import get_database_url
            dsn = get_database_url()
            if dsn:
                queue = TaskQueue(dsn)
                await queue.connect()
                _logger.info(f"TRACING [Worker] Connected to TaskQueue at {dsn.split('@')[-1]}")
        
        if not queue:
            _logger.warning("TRACING [Worker] No TaskQueue available (in-memory mode?), background processing disabled")
            return

        while True:
            try:
                # Poll for next submitted task
                task = await queue.pop()
                if task:
                    _logger.info(f"TRACING [Worker] Popped task {task.id} for processing")
                    
                    # Create a context for execution
                    # We need to bridge Task -> RequestContext -> Executor
                    from a2a.server.agent_execution import RequestContext
                    from a2a.server.events import EventQueue
                    from a2a.types import Message, Part, TextPart, Role
                    
                    # Create a dummy event queue that pushes updates back to the queue/push notifications
                    # But wait, logic is: Executor -> EventQueue -> TaskUpdater -> TaskStore
                    # We need an EventQueue that writes to our TaskStore/PushSender
                    
                    # We can use our exiting push_sender logic by creating a custom event queue
                    # or by manually handling updates.
                    
                    # The Executor expects an EventQueue.
                    # a2a.server.events.InMemoryEventQueue is simple but doesn't persist.
                    # We need to ensure updates go to DB and Push Notifications.
                    
                    # Ideally we use the same mechanism as the HTTP handler.
                    # Let's construct a context and run it.
                    
                    # Extract the user message from the task history
                    # The task object from queue should have the initial message
                    # But 'task.status.message' is the *status* message, not necessarily the input.
                    # A2A tasks don't store the *input* message directly on the task object 
                    # except maybe in history or context.
                    
                    # However, new_task(message) was called. Task ID is linked.
                    # If the task was created via send_message(..., task_id=None), 
                    # the Message object was passed.
                    
                    # When we pushed to queue, we stored the Task object.
                    # Wait, the Task object *itself* doesn't strictly contain the input prompt 
                    # unless it was added to the history or strictly defined.
                    
                    # Let's look at how connection.py sends it:
                    # message = Message(..., task_id=None)
                    # client.send_message(message)
                    # The server receives 'message'. 
                    # If blocking=False, DefaultRequestHandler calls 'new_task(message)'
                    # and saves it.
                    
                    # Does 'new_task' save the message content? 
                    # Usually it sets the task description or first history item.
                    # Let's assume we can recover the input from the task data or we treat this
                    # as a limitation and fix connection.py to explicit send input.
                    
                    # Actually, A2A Task object has 'description' or we check 
                    # if we can find the message in the task store?
                    # No, the task store only stores the Task.
                    
                    # workaround: The input message might be lost if not stored in task!
                    # BUT, for now, let's assume we can't easily get the *original* message content
                    # if it's not on the Task object. 
                    # Checking a2a.types.Task: has 'id', 'context_id', 'status', 'description', 'artifacts', 'sub_tasks'.
                    
                    # If 'description' is set to the prompt, we are good.
                    # If not, we might be executing an empty task.
                    
                    # A2A Task objects might not have a 'description' attribute directly 
                    # depending on the SDK version. We try to get it safely or fallback to history.
                    query = getattr(task, "description", None)
                    
                    if not query and task.history:
                        # Try to find the first user message in history
                        for msg in task.history:
                            if msg.role == Role.user:
                                # Extract text from parts
                                text_parts = []
                                for part in msg.parts:
                                    if hasattr(part.root, "text"):
                                        text_parts.append(part.root.text)
                                if text_parts:
                                    query = " ".join(text_parts)
                                    break
                    
                    if not query:
                        query = "Process task"
                    
                    from uuid import uuid4
                    message = Message(
                        message_id=str(uuid4()),
                        role=Role.user,
                        parts=[Part(root=TextPart(text=query))],
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                    
                    context = RequestContext(
                        message=message,
                        current_task=task,
                    )
                    
                    # Custom EventQueue that delegates to our TaskStore and PushSender
                    class WorkerEventQueue(EventQueue):
                        def __init__(self, store, sender):
                            self.store = store
                            self.sender = sender
                            
                        async def enqueue_event(self, event: Any) -> None:
                            # We only care about Task events for updates
                            from a2a.types import Task
                            if isinstance(event, Task):
                                # Update DB
                                await self.store.update_task(event)
                                # Send Push Notification
                                try:
                                    if self.sender:
                                        await self.sender.send_notification(event)
                                except Exception as e:
                                    _logger.error(f"TRACING [Worker] Push failed: {e}")

                    event_queue = WorkerEventQueue(self.task_store, self.push_sender)
                    
                    # Execute!
                    _logger.info(f"TRACING [Worker] Executing task {task.id} with prompt length {len(query)}")
                    
                    # Fire and forget execution? No, we should await it to not block the worker 
                    # but we want concurrency. 
                    # Ideally we spawn a task for execution so the worker can pop the next one.
                    asyncio.create_task(self.executor.execute(context, event_queue))
                    
                else:
                    # Queue empty, wait a bit
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                _logger.error(f"TRACING [Worker] Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)

    def get_agent_card(self) -> AgentCard:
        """Get the agent card for A2A discovery.

        Returns:
            The AgentCard for this server.
        """
        base_url = f"http://{self.host}:{self.port}"
        if isinstance(self.agent, (RootAgent, LeadAgent)):
            return self.agent.get_agent_card(base_url)
        return self.agent.get_agent_card(base_url)

    async def _handle_health(self, request: Request) -> Response:
        """Handle health check requests."""
        return JSONResponse({"status": "ok"})

    async def _handle_agent_card(self, request: Request) -> Response:
        """Handle agent card discovery requests."""
        return JSONResponse(self.get_agent_card().model_dump(exclude_none=True))

    async def _handle_agents_status(self, request: Request) -> Response:
        """Handle agent hierarchy status requests."""
        if isinstance(self.agent, RootAgent):
            status = await self.agent.get_hierarchy_status()
            return JSONResponse(status)
        return JSONResponse({
            "root": {"name": self.agent_name, "healthy": True},
            "sub_agents": {},
        })

    async def _handle_members_status(self, request: Request) -> Response:
        """Handle team member status requests (for lead agents)."""
        if isinstance(self.agent, LeadAgent):
            status = await self.agent.get_member_status()
            return JSONResponse({
                "lead": self.agent_name,
                "members": status,
            })
        return JSONResponse({"error": "not a lead agent"}, status_code=400)

    async def _handle_task_notification(self, request: Request) -> Response:
        """Handle incoming push notifications for task updates.

        This endpoint receives task updates from agents we've delegated to.
        The task is matched to pending tasks and callbacks are invoked.
        """
        try:
            data = await request.json()

            # Verify token if provided
            token = request.headers.get("X-A2A-Notification-Token", "")
            task_id = data.get("id", "")

            # Basic validation - token should match task ID if we set it that way
            if token and token != task_id:
                _logger.warning(f"Token mismatch for task notification: {token} != {task_id}")

            # Parse the task from the notification
            from a2a.types import Task
            task = Task.model_validate(data)
            _logger.info(f"TRACING [Webhook] Received task update: id='{task.id}' state='{task.status.state}'")

            # Handle the notification
            handled = await handle_task_notification(task)

            if handled:
                _logger.info(f"TRACING [Webhook] Handled task notification: {task.id} state={task.status.state}")
                return JSONResponse({"status": "ok", "handled": True})
            else:
                _logger.warning(f"TRACING [Webhook] Unknown task notification (duplicate?): {task.id}")
                return JSONResponse({"status": "ok", "handled": False})

        except Exception as e:
            _logger.error(f"TRACING [Webhook] Error handling task notification: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=400)

    async def _handle_pending_tasks(self, request: Request) -> Response:
        """Get list of pending async tasks."""
        pending_manager = get_pending_task_manager()
        pending = pending_manager.list_pending()

        # Also get delegated tasks from agent if applicable
        delegated = {}
        if isinstance(self.agent, RootAgent):
            delegated = self.agent.get_delegated_tasks()
        elif isinstance(self.agent, LeadAgent):
            delegated = self.agent.get_delegated_tasks()

        return JSONResponse({
            "pending_notifications": [
                {
                    "task_id": p.task_id,
                    "agent_name": p.agent_name,
                    "context_id": p.context_id,
                    "created_at": p.created_at,
                }
                for p in pending
            ],
            "delegated_tasks": delegated,
        })

    async def _handle_agents_details(self, request: Request) -> Response:
        """Handle individual agent detail requests."""
        name = request.query_params.get("name", "")
        if not name:
            return JSONResponse({"error": "name parameter required"}, status_code=400)

        if isinstance(self.agent, RootAgent):
            # Try to load the persona directly by type name (e.g., "product_owner")
            try:
                persona = self.agent.persona_loader.load(name)
                # Check health by querying the agent's URL
                healthy = False
                error = ""
                if persona.url:
                    try:
                        response = await self.agent.httpx_client.get(
                            f"{persona.url}/health",
                            timeout=5.0,
                        )
                        healthy = response.status_code == 200
                        if not healthy:
                            error = f"HTTP {response.status_code}"
                    except Exception as e:
                        error = str(e)
                else:
                    error = "no URL configured"

                return JSONResponse({
                    "name": persona.name,
                    "type": persona.type,
                    "description": persona.description,
                    "url": persona.url or "",
                    "healthy": healthy,
                    "error": error,
                    "model": persona.model,
                    "tools": persona.tools,
                })
            except FileNotFoundError:
                pass  # Persona not found by type, continue to fallback

            # Fallback: search all personas by display name
            for persona_name in self.agent.persona_loader.get_names():
                try:
                    persona = self.agent.persona_loader.load(persona_name)
                    if persona.name == name or persona.type == name:
                        return JSONResponse({
                            "name": persona.name,
                            "type": persona.type,
                            "description": persona.description,
                            "url": persona.url or "",
                            "healthy": False,  # Can't verify without URL check
                            "error": "",
                            "model": persona.model,
                            "tools": persona.tools,
                        })
                except Exception:
                    pass

        return JSONResponse({"error": "agent not found"}, status_code=404)

    def build_app(self) -> Starlette:
        """Build the Starlette application.

        Returns:
            The configured Starlette app.
        """
        if self._app is not None:
            return self._app

        # Use shared task store if provided
        if self.task_store is None:
            from a2a.server.tasks import InMemoryTaskStore
            _logger.warning(f"No shared task store for {self.agent_name}, using in-memory")
            self.task_store = InMemoryTaskStore()
        else:
            _logger.info(f"Using shared task store for {self.agent_name}")

        # Create A2A request handler with push notification support
        request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=self.task_store,
            push_config_store=self.push_config_store,
            push_sender=self.push_sender,
        )
        _logger.info(f"Push notifications enabled for {self.agent_name}")

        # Create A2A server
        a2a_server = A2AStarletteApplication(
            agent_card=self.get_agent_card(),
            http_handler=request_handler,
        )

        # Get A2A routes
        a2a_app = a2a_server.build()

        # Custom routes
        routes = [
            Route("/health", self._handle_health, methods=["GET"]),
            Route("/.well-known/agent-card.json", self._handle_agent_card, methods=["GET"]),
            Route("/agents/status", self._handle_agents_status, methods=["GET"]),
            Route("/agents/details", self._handle_agents_details, methods=["GET"]),
            Route("/members/status", self._handle_members_status, methods=["GET"]),
            # Push notification webhook for async task updates
            Route("/webhook/task", self._handle_task_notification, methods=["POST"]),
            # Pending tasks endpoint
            Route("/tasks/pending", self._handle_pending_tasks, methods=["GET"]),
        ]

        # Combine with A2A routes
        app = Starlette(routes=routes)

        # Mount A2A handlers
        app.mount("/", a2a_app)

        self._app = app
        return app

    def run(self) -> None:
        """Run the server synchronously."""
        app = self.build_app()
        uvicorn.run(app, host=self.host, port=self.port)

    async def run_async(self) -> None:
        """Run the server asynchronously."""
        app = self.build_app()
        config = uvicorn.Config(app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
        
        # Cleanup
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def run_async_with_worker(self) -> None:
        """Run server and start background worker."""
        self._worker_task = asyncio.create_task(self._run_background_worker())
        await self.run_async()


async def create_root_server(
    host: str = "0.0.0.0",
    port: int = 9000,
    team_config: str = "default",
) -> AgentServer:
    """Create and return a root agent server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        team_config: Team configuration name.

    Returns:
        The configured AgentServer.
    """
    async with httpx.AsyncClient(timeout=A2A_TIMEOUT) as client:
        # Construct internal webhook URL for A2A callbacks
        # Root agent service name is 'root-agent' in docker-compose
        webhook_port = os.environ.get("PORT", "8001")
        webhook_url = f"http://root-agent:{webhook_port}"
        
        _logger.info(f"Configuring RootAgent with webhook URL: {webhook_url}")
        root = RootAgent(client, team_config, webhook_base_url=webhook_url)
        return AgentServer(root, host, port)


def run_root_server(
    host: str = "0.0.0.0",
    port: int = 9000,
    team_config: str = "default",
) -> None:
    """Run the root agent server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        team_config: Team configuration name.
    """
    async def _run():
        # Get shared services
        task_store = await get_shared_task_store()
        session_service = await get_shared_session_service()
        _logger.info(f"Root server using task store: {type(task_store).__name__}")
        _logger.info(f"Root server using session service: {type(session_service).__name__}")

        # Set up shared LiteLLM session for connection pooling
        litellm.aclient_session = httpx.AsyncClient(timeout=LITELLM_TIMEOUT)
        _logger.info("Configured LiteLLM shared session")

        async with httpx.AsyncClient(timeout=A2A_TIMEOUT) as client:
            # Construct internal webhook URL for A2A callbacks
            webhook_port = os.environ.get("PORT", "8001")
            webhook_url = f"http://root-agent:{webhook_port}"

            root = RootAgent(client, team_config, webhook_base_url=webhook_url)
            # Explicitly initialize connections before serving
            await root._init_lead_connections()
            server = AgentServer(
                root, host, port,
                task_store=task_store,
                session_service=session_service,
            )
            await server.run_async_with_worker()

    asyncio.run(_run())


async def create_lead_server(
    persona_type: str,
    host: str = "0.0.0.0",
    port: int = 8001,
    team_config: str = "default",
) -> AgentServer:
    """Create and return a team lead server.

    Args:
        persona_type: The persona type for the lead.
        host: Host to bind to.
        port: Port to bind to.
        team_config: Team configuration name.

    Returns:
        The configured AgentServer.
    """
    task_store = await get_shared_task_store()
    session_service = await get_shared_session_service()
    async with httpx.AsyncClient(timeout=A2A_TIMEOUT) as client:
        # Construct internal webhook URL for A2A callbacks
        # Service names use hyphens, persona types use underscores
        service_name = persona_type.replace("_", "-")
        webhook_url = f"http://{service_name}:{port}"
        
        _logger.info(f"Configuring LeadAgent {persona_type} with webhook URL: {webhook_url}")
        lead = LeadAgent(persona_type, client, team_config, webhook_base_url=webhook_url)
        return AgentServer(
            lead, host, port,
            task_store=task_store,
            session_service=session_service,
        )


def run_lead_server(
    persona_type: str,
    host: str = "0.0.0.0",
    port: int = 8001,
    team_config: str = "default",
) -> None:
    """Run a team lead server.

    Args:
        persona_type: The persona type for the lead.
        host: Host to bind to.
        port: Port to bind to.
        team_config: Team configuration name.
    """
    async def _run():
        # Get shared services
        task_store = await get_shared_task_store()
        session_service = await get_shared_session_service()
        _logger.info(f"Lead server {persona_type} using task store: {type(task_store).__name__}")
        _logger.info(f"Lead server {persona_type} using session service: {type(session_service).__name__}")

        # Set up shared LiteLLM session for connection pooling
        litellm.aclient_session = httpx.AsyncClient(timeout=LITELLM_TIMEOUT)
        _logger.info("Configured LiteLLM shared session")

        async with httpx.AsyncClient(timeout=A2A_TIMEOUT) as client:
            # Construct internal webhook URL
            service_name = persona_type.replace("_", "-")
            webhook_url = f"http://{service_name}:{port}"
            
            lead = LeadAgent(persona_type, client, team_config, webhook_base_url=webhook_url)
            server = AgentServer(
                lead, host, port,
                task_store=task_store,
                session_service=session_service,
            )
            await server.run_async_with_worker()

    asyncio.run(_run())


def run_agent_server(
    persona_type: str,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    """Run a generic agent server for a persona.

    Args:
        persona_type: The persona type.
        host: Host to bind to.
        port: Port to bind to.
    """
    async def _run():
        # Get shared services
        task_store = await get_shared_task_store()
        session_service = await get_shared_session_service()
        _logger.info(f"Agent server {persona_type} using task store: {type(task_store).__name__}")
        _logger.info(f"Agent server {persona_type} using session service: {type(session_service).__name__}")

        # Set up shared LiteLLM session for connection pooling
        litellm.aclient_session = httpx.AsyncClient(timeout=LITELLM_TIMEOUT)
        _logger.info("Configured LiteLLM shared session")

        from .persona import PersonaLoader
        loader = PersonaLoader()
        config = loader.load(persona_type)
        agent = Agent(config, loader=loader)
        server = AgentServer(
            agent, host, port,
            task_store=task_store,
            session_service=session_service,
        )
        await server.run_async_with_worker()

    asyncio.run(_run())
