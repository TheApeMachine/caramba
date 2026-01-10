"""PostgreSQL-backed session service for ADK agents.

Provides persistent conversation history that survives restarts and
is shared across all agent containers.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import (
    GetSessionConfig,
    ListSessionsResponse,
)
from sqlalchemy import String, Text, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class SessionRecord(Base):
    """Database model for storing ADK sessions."""

    __tablename__ = "adk_sessions"

    # Composite key: app_name + user_id + session_id
    app_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Session data stored as JSON
    state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    events_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # Metadata
    last_update_time: Mapped[float] = mapped_column(nullable=False)


class DatabaseSessionService(BaseSessionService):
    """PostgreSQL-backed session service.

    Stores conversation history in PostgreSQL so it persists across
    container restarts and is shared across all agents.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        create_table: bool = True,
    ) -> None:
        """Initialize the session service.

        Args:
            engine: SQLAlchemy async engine.
            create_table: Whether to create the table if it doesn't exist.
        """
        self._engine = engine
        self._create_table = create_table
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        if self._create_table:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            _logger.info("Database session table initialized")

        self._initialized = True

    def _serialize_event(self, event: Event) -> dict[str, Any]:
        """Serialize an Event to JSON-compatible dict."""
        try:
            return event.model_dump(mode="json", exclude_none=True)
        except Exception as e:
            _logger.warning(f"Failed to serialize event: {e}")
            # Return minimal event data
            return {
                "id": event.id,
                "author": event.author,
                "timestamp": event.timestamp,
                "invocation_id": event.invocation_id,
            }

    def _deserialize_event(self, data: dict[str, Any]) -> Event | None:
        """Deserialize an Event from dict."""
        try:
            return Event.model_validate(data)
        except Exception as e:
            _logger.warning(f"Failed to deserialize event: {e}")
            return None

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create a new session or return existing one."""
        await self.initialize()

        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())

        now = time.time()
        state = state or {}

        async with self._session_factory() as db:
            # Check if session already exists
            result = await db.execute(
                select(SessionRecord).where(
                    SessionRecord.app_name == app_name,
                    SessionRecord.user_id == user_id,
                    SessionRecord.session_id == session_id,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Return existing session
                stored_state = json.loads(str(existing.state_json))
                stored_events_data = json.loads(str(existing.events_json))
                events = [
                    e for e in
                    (self._deserialize_event(d) for d in stored_events_data)
                    if e is not None
                ]
                return Session(
                    id=session_id,
                    app_name=app_name,
                    user_id=user_id,
                    state=stored_state,
                    events=events,
                    last_update_time=existing.last_update_time,
                )

            # Create new session
            record = SessionRecord(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state_json=json.dumps(state),
                events_json="[]",
                last_update_time=now,
            )
            db.add(record)
            await db.commit()

            _logger.debug(f"Created session: {app_name}/{user_id}/{session_id}")

            return Session(
                id=session_id,
                app_name=app_name,
                user_id=user_id,
                state=state,
                events=[],
                last_update_time=now,
            )

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Get an existing session."""
        await self.initialize()

        async with self._session_factory() as db:
            result = await db.execute(
                select(SessionRecord).where(
                    SessionRecord.app_name == app_name,
                    SessionRecord.user_id == user_id,
                    SessionRecord.session_id == session_id,
                )
            )
            record = result.scalar_one_or_none()

            if not record:
                return None

            state = json.loads(str(record.state_json))
            events_data = json.loads(str(record.events_json))

            # Apply config limits if specified
            if config and config.num_recent_events is not None:
                events_data = events_data[-config.num_recent_events:]

            events = [
                e for e in
                (self._deserialize_event(d) for d in events_data)
                if e is not None
            ]

            return Session(
                id=session_id,
                app_name=app_name,
                user_id=user_id,
                state=state,
                events=events,
                last_update_time=record.last_update_time,
            )

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        """List sessions for an app/user."""
        await self.initialize()

        async with self._session_factory() as db:
            query = select(SessionRecord).where(
                SessionRecord.app_name == app_name
            )
            if user_id:
                query = query.where(SessionRecord.user_id == user_id)

            result = await db.execute(query)
            records = result.scalars().all()

            sessions = []
            for record in records:
                state = json.loads(str(record.state_json))
                events_data = json.loads(str(record.events_json))
                events = [
                    e for e in
                    (self._deserialize_event(d) for d in events_data)
                    if e is not None
                ]
                sessions.append(Session(
                    id=record.session_id,
                    app_name=app_name,
                    user_id=record.user_id,
                    state=state,
                    events=events,
                    last_update_time=record.last_update_time,
                ))

            return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """Delete a session."""
        await self.initialize()

        async with self._session_factory() as db:
            await db.execute(
                delete(SessionRecord).where(
                    SessionRecord.app_name == app_name,
                    SessionRecord.user_id == user_id,
                    SessionRecord.session_id == session_id,
                )
            )
            await db.commit()
            _logger.debug(f"Deleted session: {app_name}/{user_id}/{session_id}")

    async def append_event(
        self,
        session: Session,
        event: Event,
    ) -> Event:
        """Append an event to a session and persist."""
        await self.initialize()

        now = time.time()

        async with self._session_factory() as db:
            result = await db.execute(
                select(SessionRecord).where(
                    SessionRecord.app_name == session.app_name,
                    SessionRecord.user_id == session.user_id,
                    SessionRecord.session_id == session.id,
                )
            )
            record = result.scalar_one_or_none()

            if not record:
                # Create session if it doesn't exist
                events_data = [self._serialize_event(event)]
                record = SessionRecord(
                    app_name=session.app_name,
                    user_id=session.user_id,
                    session_id=session.id,
                    state_json=json.dumps(session.state),
                    events_json=json.dumps(events_data),
                    last_update_time=now,
                )
                db.add(record)
            else:
                # Append to existing
                events_data = json.loads(str(record.events_json))
                events_data.append(self._serialize_event(event))

                # Limit events to prevent unbounded growth (keep last 100)
                if len(events_data) > 100:
                    events_data = events_data[-100:]

                record.events_json = json.dumps(events_data)
                record.state_json = json.dumps(session.state)
                record.last_update_time = now

            await db.commit()

        # Update in-memory session
        session.events.append(event)
        session.last_update_time = now

        return event


# Global instances
_engine: AsyncEngine | None = None
_session_service: DatabaseSessionService | None = None


def get_session_database_url() -> str | None:
    """Get the database URL for session storage."""
    # Check for explicit session store URL first
    url = os.environ.get("SESSION_STORE_DATABASE_URL")
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


async def get_shared_session_engine() -> AsyncEngine | None:
    """Get or create the shared database engine for sessions."""
    global _engine

    if _engine is not None:
        return _engine

    url = get_session_database_url()
    if not url:
        _logger.warning("No database URL configured for session store")
        return None

    try:
        _logger.info(f"Creating session database engine: {url.split('@')[-1]}")
        _engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        return _engine
    except Exception as e:
        _logger.error(f"Failed to create session database engine: {e}")
        return None


async def get_shared_session_service() -> BaseSessionService:
    """Get the shared session service.

    Returns a PostgreSQL-backed service if available, otherwise falls back
    to in-memory service.
    """
    global _session_service

    if _session_service is not None:
        return _session_service

    engine = await get_shared_session_engine()

    if engine is not None:
        try:
            service = DatabaseSessionService(
                engine=engine,
                create_table=True,
            )
            await service.initialize()
            _session_service = service
            _logger.info("Using PostgreSQL-backed session service")
            return service
        except Exception as e:
            _logger.error(f"Failed to create database session service: {e}")

    # Fall back to in-memory
    from google.adk.sessions import InMemorySessionService
    _logger.warning("Falling back to in-memory session service (sessions won't persist!)")
    return InMemorySessionService()
