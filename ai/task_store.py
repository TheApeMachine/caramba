"""Shared task store for A2A agents.

Provides a PostgreSQL-backed task store that all agents share,
enabling proper task handoff and artifact accumulation.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore, TaskStore

_logger = logging.getLogger(__name__)


def get_database_url() -> str | None:
    """Get the database URL from environment.
    
    Returns:
        The async database URL, or None if not configured.
    """
    # Check for explicit task store URL first
    url = os.environ.get("TASK_STORE_DATABASE_URL")
    if url:
        return url
    
    # Fall back to projectboard database (they can share)
    url = os.environ.get("PROJECTBOARD_DATABASE_URL")
    if url:
        # Convert to async URL if needed
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    
    # Try to construct from individual components
    host = os.environ.get("POSTGRES_HOST", "postgres-projectboard")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "projectboard")
    password = os.environ.get("POSTGRES_PASSWORD", "projectboard")
    database = os.environ.get("POSTGRES_DB", "projectboard")
    
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


_engine: AsyncEngine | None = None
_task_store: TaskStore | None = None


async def get_shared_engine() -> AsyncEngine | None:
    """Get or create the shared database engine.
    
    Returns:
        The async engine, or None if database is not available.
    """
    global _engine
    
    if _engine is not None:
        return _engine
    
    url = get_database_url()
    if not url:
        _logger.warning("No database URL configured for task store")
        return None
    
    try:
        _logger.info(f"Creating shared database engine: {url.split('@')[-1]}")  # Log without password
        _engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        return _engine
    except Exception as e:
        _logger.error(f"Failed to create database engine: {e}")
        return None


async def get_shared_task_store() -> TaskStore:
    """Get the shared task store.
    
    Returns a PostgreSQL-backed store if available, otherwise falls back
    to in-memory store.
    
    Returns:
        The shared TaskStore instance.
    """
    global _task_store
    
    if _task_store is not None:
        return _task_store
    
    engine = await get_shared_engine()
    
    if engine is not None:
        try:
            store = DatabaseTaskStore(
                engine=engine,
                create_table=True,
                table_name="a2a_tasks",  # Use distinct table name
            )
            await store.initialize()
            _task_store = store
            _logger.info("Using PostgreSQL-backed task store")
            return store
        except Exception as e:
            _logger.error(f"Failed to create database task store: {e}")
    
    # Fall back to in-memory
    _logger.warning("Falling back to in-memory task store (tasks won't be shared!)")
    _task_store = InMemoryTaskStore()
    return _task_store


def get_task_store_sync() -> TaskStore:
    """Get a task store synchronously (for initialization).
    
    This returns an InMemoryTaskStore initially, which should be replaced
    with the shared store once the async context is available.
    
    Returns:
        A TaskStore instance.
    """
    global _task_store
    if _task_store is not None:
        return _task_store
    return InMemoryTaskStore()
