"""Push notification support for A2A async operations.

Implements the PushNotificationConfigStore and PushNotificationSender
interfaces for proper A2A async task handling.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from a2a.server.tasks import PushNotificationConfigStore, PushNotificationSender
from a2a.types import (
    PushNotificationConfig,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskStatusUpdateEvent,
)

_logger = logging.getLogger(__name__)


def _extract_credential_token(cred: Any) -> str | None:
    """Normalize a credential representation to a bearer token string.

    The A2A SDKs may represent credentials as:
    - a string token
    - a dict with keys "key" or "token"
    - an object with attributes "key" or "token"
    """
    if cred is None:
        return None
    if isinstance(cred, str):
        return cred
    if isinstance(cred, dict):
        val = cred.get("key") or cred.get("token")
        return str(val) if val else None

    val = getattr(cred, "key", None) or getattr(cred, "token", None)
    return str(val) if val else None


class InMemoryPushNotificationConfigStore(PushNotificationConfigStore):
    """In-memory storage for push notification configurations."""

    def __init__(self) -> None:
        """Initialize the store."""
        self._configs: dict[str, list[PushNotificationConfig]] = {}

    async def set_info(
        self, task_id: str, notification_config: PushNotificationConfig
    ) -> None:
        """Store a push notification config for a task."""
        if task_id not in self._configs:
            self._configs[task_id] = []
        
        # Check if config with same URL already exists
        for i, existing in enumerate(self._configs[task_id]):
            if existing.url == notification_config.url:
                self._configs[task_id][i] = notification_config
                return
        
        self._configs[task_id].append(notification_config)
        _logger.debug(f"Stored push config for task {task_id}: {notification_config.url}")

    async def get_info(self, task_id: str) -> list[PushNotificationConfig]:
        """Get push notification configs for a task."""
        return self._configs.get(task_id, [])

    async def delete_info(
        self, task_id: str, config_id: str | None = None
    ) -> None:
        """Delete push notification config(s) for a task."""
        if config_id is None:
            self._configs.pop(task_id, None)
        else:
            if task_id in self._configs:
                self._configs[task_id] = [
                    c for c in self._configs[task_id]
                    if getattr(c, "id", None) != config_id
                ]
        _logger.debug(f"Deleted push config for task {task_id}")


class HttpPushNotificationSender(PushNotificationSender):
    """Sends push notifications via HTTP POST to configured webhooks."""

    def __init__(
        self,
        config_store: PushNotificationConfigStore,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the sender.
        
        Args:
            config_store: The store to look up notification configs.
            http_client: Optional httpx client for making requests.
        """
        self._config_store = config_store
        self._client = http_client
        self._owns_client = http_client is None

    async def send_notification(self, task: Task) -> None:
        """Send push notification for a task update.
        
        Per A2A protocol, sends an HTTP POST with a StreamResponse body
        containing the task update.
        """
        configs = await self._config_store.get_info(task.id)
        if not configs:
            _logger.debug(f"No push configs for task {task.id}")
            return

        # Build the notification payload per A2A spec
        # The payload is a SendStreamingMessageSuccessResponse with the task
        payload = SendStreamingMessageSuccessResponse(
            result=TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=task.status,
                final=task.status.state.value in ("completed", "failed", "canceled"),
            )
        )
        payload_dict = payload.model_dump(exclude_none=True, mode="json")

        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=30.0)

        try:
            for config in configs:
                try:
                    headers: dict[str, str] = {
                        "Content-Type": "application/json",
                    }
                    
                    # Add token if provided (for client verification)
                    if config.token:
                        headers["X-A2A-Token"] = config.token

                    # Add authentication if configured
                    if config.authentication:
                        for cred in config.authentication.credentials or []:
                            token = _extract_credential_token(cred)
                            if token is not None:
                                headers["Authorization"] = f"Bearer {token}"
                                break

                    _logger.info(
                        f"Sending push notification for task {task.id} "
                        f"to {config.url} (state: {task.status.state.value})"
                    )

                    response = await client.post(
                        config.url,
                        json=payload_dict,
                        headers=headers,
                    )
                    
                    if response.status_code >= 400:
                        _logger.warning(
                            f"Push notification failed: {response.status_code} "
                            f"for task {task.id} to {config.url}"
                        )
                    else:
                        _logger.debug(
                            f"Push notification sent: {response.status_code} "
                            f"for task {task.id}"
                        )

                except Exception as e:
                    _logger.error(
                        f"Error sending push notification for task {task.id} "
                        f"to {config.url}: {e}"
                    )
        finally:
            if self._owns_client and client:
                await client.aclose()
