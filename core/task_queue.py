"""Task queue implementation using Postgres."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

import asyncpg
from a2a.types import (
    Task,
    TaskState,
    TaskStatus,
)

_logger = logging.getLogger(__name__)


class TaskQueue:
    """Asynchronous task queue backed by Postgres."""

    def __init__(self, dsn: str) -> None:
        """Initialize the task queue.

        Args:
            dsn: Postgres connection string.
        """
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Connect to the database and initialize schema."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn)
            await self._init_schema()

    async def close(self) -> None:
        """Close the database connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _init_schema(self) -> None:
        """Initialize the database schema."""
        if not self._pool:
            raise RuntimeError("TaskQueue not connected")

        async with self._pool.acquire() as conn:
            # We store the full A2A Task object as JSONB
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    task_data JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
                """
            )

    async def push(self, task: Task) -> str:
        """Push a new task to the queue.

        Args:
            task: The A2A Task object.

        Returns:
            The Task ID.
        """
        if not self._pool:
            raise RuntimeError("TaskQueue not connected")

        now = datetime.now()
        
        # Ensure task is in SUBMITTED state if not specified
        if task.status.state == TaskState.unknown:
            task.status.state = TaskState.submitted

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tasks (id, context_id, state, task_data, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                task.id,
                task.context_id,
                task.status.state.name,
                json.dumps(task.model_dump(mode="json")),
                now,
                now,
            )
        
        _logger.info(f"Pushed task {task.id} (state={task.status.state.name})")
        return task.id

    async def pop(self) -> Task | None:
        """Pop a pending task from the queue and mark it as processing.
        
        Finds tasks in SUBMITTED state.

        Returns:
            The A2A Task to process, or None if queue is empty.
        """
        if not self._pool:
            raise RuntimeError("TaskQueue not connected")

        async with self._pool.acquire() as conn:
            # Atomic update to claim a task
            # We look for SUBMITTED tasks to move to WORKING
            row = await conn.fetchrow(
                """
                UPDATE tasks
                SET state = $1, updated_at = $2, 
                    task_data = jsonb_set(
                        task_data, 
                        '{status,state}', 
                        to_jsonb($3::text)
                    )
                WHERE id = (
                    SELECT id
                    FROM tasks
                    WHERE state = $4
                    ORDER BY created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING task_data
                """,
                TaskState.working.name,
                datetime.now(),
                TaskState.working.name, # For JSON update
                TaskState.submitted.name,
            )
            
            if row:
                return Task.model_validate(json.loads(row["task_data"]))
            return None

    async def update(self, task: Task) -> None:
        """Update an existing task.
        
        Args:
            task: The updated A2A Task object.
        """
        if not self._pool:
            raise RuntimeError("TaskQueue not connected")

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE tasks
                SET state = $1, task_data = $2, updated_at = $3
                WHERE id = $4
                """,
                task.status.state.name,
                json.dumps(task.model_dump(mode="json")),
                datetime.now(),
                task.id,
            )
        _logger.info(f"Updated task {task.id} (state={task.status.state.name})")

    async def get_status(self, task_id: str) -> Task | None:
        """Get the status of a task.

        Args:
            task_id: The ID of the task.

        Returns:
            The A2A Task, or None if not found.
        """
        if not self._pool:
            raise RuntimeError("TaskQueue not connected")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT task_data FROM tasks WHERE id = $1",
                task_id,
            )
            if row:
                return Task.model_validate(json.loads(row["task_data"]))
            return None

