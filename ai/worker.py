"""Background worker for A2A asynchronous tasks

Since we work fully async, without blocking, we need a background worker
to poll for pending tasks. Each agent will have its own task queue, and
the worker will poll the queue and pull the tasks for the agent to work on.
"""
from __future__ import annotations

import asyncio
import logging

from caramba.ai.task.store import TaskStore
from caramba.console import logger


class TaskWorker:
    """Background worker for A2A asynchronous tasks

    We set up the worker with the task store so it can periodically poll
    for pending tasks and process them.
    """
    def __init__(self, task_store: TaskStore, tenant: str) -> None:
        logger.trace("Initializing TaskWorker")
        self.tenant = tenant
        self.task_store = task_store
        self.queue = asyncio.Queue()

    async def run(self) -> None:
        """Run the background worker."""
        logger.trace("Running TaskWorker")

        while True:
            await asyncio.sleep(1)
            logger.trace("Polling for pending tasks")
            tasks = [t async for t in self.task_store.list_tasks(tenant=self.tenant)]
            logger.trace(f"Found {len(tasks)} pending tasks")

            for task in tasks:
                logger.trace(f"Enqueuing task {task.id}")
                await self.queue.put(task)