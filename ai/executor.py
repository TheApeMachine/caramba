"""A2A Agent Executor.

Bridges ADK agents with the A2A protocol by handling task execution,
status updates, and streaming responses.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InvalidParamsError,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event as ADKEvent
from google.genai import types



_logger = logging.getLogger(__name__)


class ADKAgentExecutor(AgentExecutor):
    """Executor that runs ADK agents and converts to A2A format.

    Handles the lifecycle of task execution including status updates,
    streaming responses, and artifact generation.
    """

    def __init__(
        self,
        runner: Runner,
        agent_name: str,
        user_id: str = "default_user",
    ) -> None:
        """Initialize the executor.

        Args:
            runner: The ADK runner instance.
            agent_name: Name of the agent being executed.
            user_id: Default user ID for sessions.
        """
        self.runner = runner
        self.agent_name = agent_name
        self.user_id = user_id
        self._session_cache: set[str] = set()

    async def _ensure_session(self, session_id: str) -> None:
        """Ensure a session exists, creating it if necessary.

        Args:
            session_id: The session ID to ensure exists.
        """
        if session_id in self._session_cache:
            return

        # Try to get existing session
        session = await self.runner.session_service.get_session(
            app_name="caramba",
            user_id=self.user_id,
            session_id=session_id,
        )

        if session is None:
            # Create new session
            await self.runner.session_service.create_session(
                app_name="caramba",
                user_id=self.user_id,
                session_id=session_id,
            )

        self._session_cache.add(session_id)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a task from an A2A request.

        Args:
            context: The request context containing the message.
            event_queue: Queue for sending events back to the caller.

        Raises:
            ServerError: If the request is invalid.
        """
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        message = context.message

        if not task:
            if message is None:
                raise ServerError(error=InvalidParamsError())
            task = new_task(message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        session_id = task.context_id or "default_session"

        # Ensure session exists before running
        await self._ensure_session(session_id)

        content = types.Content(role="user", parts=[types.Part(text=query)])

        try:
            run_config = RunConfig(max_llm_calls=50)
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=content,
                run_config=run_config,
            ):
                await self._process_adk_event(event, updater, task)

                if event.is_final_response():
                    await self._handle_final_response(event, updater, task)
                    return

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"Error: {str(e)}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )

    async def _process_adk_event(
        self,
        event: ADKEvent,
        updater: TaskUpdater,
        task: Task,
    ) -> None:
        """Process an ADK event and send status updates.

        Args:
            event: The ADK event.
            updater: The task updater.
            task: The current task.
        """
        # Check for function calls (tool usage)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    # Notify about tool call
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"Calling tool: {part.function_call.name}",
                            task.context_id,
                            task.id,
                        ),
                    )

    async def _handle_final_response(
        self,
        event: ADKEvent,
        updater: TaskUpdater,
        task: Task,
    ) -> None:
        """Handle the final response from ADK.

        Args:
            event: The final ADK event.
            updater: The task updater.
            task: The current task.
        """
        if not event.content or not event.content.parts:
            await updater.complete()
            return

        # Collect all response parts
        text_parts = []
        data_parts = []

        for part in event.content.parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_response:
                # Convert function response to data
                data_parts.append(part.function_response.response)

        # Create artifact from response
        artifact_parts = []
        if text_parts:
            combined_text = "\n".join(text_parts)
            artifact_parts.append(TextPart(text=combined_text))

        for data in data_parts:
            artifact_parts.append(DataPart(data=data))

        if artifact_parts:
            await updater.add_artifact(
                artifact_parts,
                name=f"{self.agent_name}-result",
            )

        await updater.complete()

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate the request. Returns True if INVALID.

        Args:
            context: The request context.

        Returns:
            True if the request is invalid.
        """
        return not context.get_user_input()

    async def cancel(
        self,
        request: RequestContext,
        event_queue: EventQueue,
    ) -> Task | None:
        """Cancel a running task.

        Args:
            request: The cancel request context.
            event_queue: The event queue.

        Raises:
            ServerError: Cancellation not supported.
        """
        raise ServerError(error=UnsupportedOperationError())


class StreamingExecutor(ADKAgentExecutor):
    """Executor with enhanced streaming support.

    Provides real-time token streaming for long-running responses.
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute with streaming support.

        Args:
            context: The request context.
            event_queue: The event queue for streaming.
        """
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        message = context.message
        
        _logger.info(f"TRACING [Executor] Starting execution. Query preview: {query[:100]}...")

        if not task:
            if message is None:
                raise ServerError(error=InvalidParamsError())
            task = new_task(message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        session_id = task.context_id or "default_session"

        # Ensure session exists before running
        await self._ensure_session(session_id)

        content = types.Content(role="user", parts=[types.Part(text=query)])

        accumulated_text: list[str] = []
        
        _logger.info(f"TRACING [Executor] user_id='{self.user_id}' session_id='{session_id}' - calling runner.run_async")

        try:
            run_config = RunConfig(
                max_llm_calls=50,
                streaming_mode=StreamingMode.SSE,
            )
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=content,
                run_config=run_config,
            ):
                # Stream intermediate content
                if event.content and event.content.parts and not event.is_final_response():
                    for part in event.content.parts:
                        if part.text:
                            accumulated_text.append(part.text)
                            # Send streaming update
                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message(
                                    part.text,
                                    task.context_id,
                                    task.id,
                                ),
                            )

                if event.is_final_response():
                    # If we already streamed content, just complete without re-sending
                    if accumulated_text:
                        await updater.complete()
                    else:
                        # No streaming happened, send final response normally
                        await self._handle_final_response(event, updater, task)
                    return

        except Exception as e:
            _logger.error(f"TRACING [Executor] Execution error: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"Error: {str(e)}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
