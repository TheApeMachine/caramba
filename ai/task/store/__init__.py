"""Task store for A2A agents.

This is a persistent store, specifically defined in the A2A spec.
"""
from __future__ import annotations
from typing import AsyncGenerator

from a2a.types import Task

from a2a.server.tasks.database_task_store import DatabaseTaskStore
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import select

from caramba.console import logger


class TaskStore(DatabaseTaskStore):
    """Task store for A2A agents.

    This is a persistent store, specifically defined in the A2A spec. It
    Inherits from the DatabaseTaskStore class from the A2A server package,
    which is a SQLAlchemy-based implementation of the TaskStore interface.
    """
    def __init__(
        self,
        engine: AsyncEngine,
        create_table: bool = True,
        table_name: str = 'tasks',
    ) -> None:
        """Initialize the task store.

        Args:
            engine: The SQLAlchemy engine to use.
            create_table: Whether to create the table if it doesn't exist.
            table_name: The name of the table to use.
        """
        logger.trace("Initializing TaskStore")

        super().__init__(
            engine,
            create_table,
            table_name
        )

    async def list_tasks(self, tenant: str) -> AsyncGenerator[Task, None]:
        """List tasks for a tenant

        This is ultimately used by an Agent to get a new task
        from its task queue.
        """
        async with self.engine.connect() as connect:
            result = await connect.execute(
                select(self.task_model).where(self.task_model.context_id == tenant)
            )
            for task in result.scalars().all():
                yield self._from_orm(task)