"""PostgreSQL storage for project management data."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import asyncpg
from asyncpg import Pool

from .models import Epic, Priority, Project, Status, Story, Task

logger = logging.getLogger(__name__)


class ProjectBoardStorage:
    """PostgreSQL storage for project board data."""

    def __init__(self, pool: Pool):
        """Initialize with an asyncpg connection pool."""
        self.pool = pool

    @classmethod
    async def create(cls, database_url: str) -> ProjectBoardStorage:
        """Create a new storage instance and initialize the database."""
        pool = await asyncpg.create_pool(database_url, min_size=1, max_size=10)
        storage = cls(pool)
        await storage._init_schema()
        return storage

    async def close(self) -> None:
        """Close the connection pool."""
        await self.pool.close()

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    objectives JSONB,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    tags JSONB
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS epics (
                    id UUID PRIMARY KEY,
                    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    description TEXT,
                    goals JSONB,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    depends_on JSONB,
                    tags JSONB
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    id UUID PRIMARY KEY,
                    epic_id UUID NOT NULL REFERENCES epics(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    description TEXT,
                    acceptance_criteria JSONB,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    assignee TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    depends_on JSONB,
                    tags JSONB
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id UUID PRIMARY KEY,
                    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    assignee TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    depends_on JSONB,
                    tags JSONB
                )
            """)

            # Create indexes for common queries
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_epics_project ON epics(project_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_stories_epic ON stories(epic_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_story ON tasks(story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_epics_status ON epics(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_stories_status ON stories(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")

    # Project CRUD
    async def create_project(self, project: Project) -> Project:
        """Create a new project."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO projects (id, title, description, objectives, status, created_at, updated_at, completed_at, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                project.id,
                project.title,
                project.description,
                json.dumps(project.objectives),
                project.status.value,
                project.created_at,
                project.updated_at,
                project.completed_at,
                json.dumps(project.tags),
            )
        return project

    async def get_project(self, project_id: UUID) -> Optional[Project]:
        """Get a project by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM projects WHERE id = $1", project_id)
            if not row:
                return None
            return Project(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                objectives=json.loads(row["objectives"]),
                status=Status(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                tags=json.loads(row["tags"]),
            )

    async def list_projects(self, status: Optional[Status] = None) -> list[Project]:
        """List all projects, optionally filtered by status."""
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch("SELECT * FROM projects WHERE status = $1 ORDER BY created_at DESC", status.value)
            else:
                rows = await conn.fetch("SELECT * FROM projects ORDER BY created_at DESC")

            return [
                Project(
                    id=row["id"],
                    title=row["title"],
                    description=row["description"],
                    objectives=json.loads(row["objectives"]),
                    status=Status(row["status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    completed_at=row["completed_at"],
                    tags=json.loads(row["tags"]),
                )
                for row in rows
            ]

    async def update_project(self, project: Project) -> Project:
        """Update a project."""
        project.updated_at = datetime.utcnow()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE projects
                SET title = $2, description = $3, objectives = $4, status = $5,
                    updated_at = $6, completed_at = $7, tags = $8
                WHERE id = $1
                """,
                project.id,
                project.title,
                project.description,
                json.dumps(project.objectives),
                project.status.value,
                project.updated_at,
                project.completed_at,
                json.dumps(project.tags),
            )
        return project

    # Epic CRUD
    async def create_epic(self, epic: Epic) -> Epic:
        """Create a new epic."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO epics (id, project_id, title, description, goals, status, priority,
                                   created_at, updated_at, completed_at, depends_on, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                epic.id,
                epic.project_id,
                epic.title,
                epic.description,
                json.dumps(epic.goals),
                epic.status.value,
                epic.priority.value,
                epic.created_at,
                epic.updated_at,
                epic.completed_at,
                json.dumps([str(d) for d in epic.depends_on]),
                json.dumps(epic.tags),
            )
        return epic

    async def list_epics(self, project_id: Optional[UUID] = None, status: Optional[Status] = None) -> list[Epic]:
        """List epics, optionally filtered by project and/or status."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM epics WHERE 1=1"
            params = []
            if project_id:
                params.append(project_id)
                query += f" AND project_id = ${len(params)}"
            if status:
                params.append(status.value)
                query += f" AND status = ${len(params)}"
            query += " ORDER BY created_at DESC"

            rows = await conn.fetch(query, *params)
            return [
                Epic(
                    id=row["id"],
                    project_id=row["project_id"],
                    title=row["title"],
                    description=row["description"],
                    goals=json.loads(row["goals"]),
                    status=Status(row["status"]),
                    priority=Priority(row["priority"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    completed_at=row["completed_at"],
                    depends_on=[UUID(d) for d in json.loads(row["depends_on"])],
                    tags=json.loads(row["tags"]),
                )
                for row in rows
            ]

    async def update_epic(self, epic: Epic) -> Epic:
        """Update an epic."""
        epic.updated_at = datetime.utcnow()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE epics
                SET title = $2, description = $3, goals = $4, status = $5, priority = $6,
                    updated_at = $7, completed_at = $8, depends_on = $9, tags = $10
                WHERE id = $1
                """,
                epic.id,
                epic.title,
                epic.description,
                json.dumps(epic.goals),
                epic.status.value,
                epic.priority.value,
                epic.updated_at,
                epic.completed_at,
                json.dumps([str(d) for d in epic.depends_on]),
                json.dumps(epic.tags),
            )
        return epic

    # Story CRUD
    async def create_story(self, story: Story) -> Story:
        """Create a new story."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stories (id, epic_id, title, description, acceptance_criteria, status, priority,
                                     assignee, created_at, updated_at, completed_at, depends_on, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                story.id,
                story.epic_id,
                story.title,
                story.description,
                json.dumps(story.acceptance_criteria),
                story.status.value,
                story.priority.value,
                story.assignee,
                story.created_at,
                story.updated_at,
                story.completed_at,
                json.dumps([str(d) for d in story.depends_on]),
                json.dumps(story.tags),
            )
        return story

    async def list_stories(self, epic_id: Optional[UUID] = None, status: Optional[Status] = None) -> list[Story]:
        """List stories, optionally filtered by epic and/or status."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM stories WHERE 1=1"
            params = []
            if epic_id:
                params.append(epic_id)
                query += f" AND epic_id = ${len(params)}"
            if status:
                params.append(status.value)
                query += f" AND status = ${len(params)}"
            query += " ORDER BY created_at DESC"

            rows = await conn.fetch(query, *params)
            return [
                Story(
                    id=row["id"],
                    epic_id=row["epic_id"],
                    title=row["title"],
                    description=row["description"],
                    acceptance_criteria=json.loads(row["acceptance_criteria"]),
                    status=Status(row["status"]),
                    priority=Priority(row["priority"]),
                    assignee=row["assignee"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    completed_at=row["completed_at"],
                    depends_on=[UUID(d) for d in json.loads(row["depends_on"])],
                    tags=json.loads(row["tags"]),
                )
                for row in rows
            ]

    async def update_story(self, story: Story) -> Story:
        """Update a story."""
        story.updated_at = datetime.utcnow()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE stories
                SET title = $2, description = $3, acceptance_criteria = $4, status = $5, priority = $6,
                    assignee = $7, updated_at = $8, completed_at = $9, depends_on = $10, tags = $11
                WHERE id = $1
                """,
                story.id,
                story.title,
                story.description,
                json.dumps(story.acceptance_criteria),
                story.status.value,
                story.priority.value,
                story.assignee,
                story.updated_at,
                story.completed_at,
                json.dumps([str(d) for d in story.depends_on]),
                json.dumps(story.tags),
            )
        return story

    # Task CRUD
    async def create_task(self, task: Task) -> Task:
        """Create a new task."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tasks (id, story_id, title, description, status, priority,
                                   assignee, created_at, updated_at, completed_at, depends_on, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                task.id,
                task.story_id,
                task.title,
                task.description,
                task.status.value,
                task.priority.value,
                task.assignee,
                task.created_at,
                task.updated_at,
                task.completed_at,
                json.dumps([str(d) for d in task.depends_on]),
                json.dumps(task.tags),
            )
        return task

    async def list_tasks(self, story_id: Optional[UUID] = None, status: Optional[Status] = None) -> list[Task]:
        """List tasks, optionally filtered by story and/or status."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM tasks WHERE 1=1"
            params = []
            if story_id:
                params.append(story_id)
                query += f" AND story_id = ${len(params)}"
            if status:
                params.append(status.value)
                query += f" AND status = ${len(params)}"
            query += " ORDER BY created_at DESC"

            rows = await conn.fetch(query, *params)
            return [
                Task(
                    id=row["id"],
                    story_id=row["story_id"],
                    title=row["title"],
                    description=row["description"],
                    status=Status(row["status"]),
                    priority=Priority(row["priority"]),
                    assignee=row["assignee"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    completed_at=row["completed_at"],
                    depends_on=[UUID(d) for d in json.loads(row["depends_on"])],
                    tags=json.loads(row["tags"]),
                )
                for row in rows
            ]

    async def update_task(self, task: Task) -> Task:
        """Update a task."""
        task.updated_at = datetime.utcnow()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE tasks
                SET title = $2, description = $3, status = $4, priority = $5,
                    assignee = $6, updated_at = $7, completed_at = $8, depends_on = $9, tags = $10
                WHERE id = $1
                """,
                task.id,
                task.title,
                task.description,
                task.status.value,
                task.priority.value,
                task.assignee,
                task.updated_at,
                task.completed_at,
                json.dumps([str(d) for d in task.depends_on]),
                json.dumps(task.tags),
            )
        return task

    async def get_project_hierarchy(self, project_id: UUID) -> dict:
        """Get full project hierarchy (project > epics > stories > tasks)."""
        project = await self.get_project(project_id)
        if not project:
            return {}

        epics = await self.list_epics(project_id=project_id)
        result = {
            "project": project.model_dump(mode="json"),
            "epics": []
        }

        for epic in epics:
            stories = await self.list_stories(epic_id=epic.id)
            epic_data = {
                "epic": epic.model_dump(mode="json"),
                "stories": []
            }

            for story in stories:
                tasks = await self.list_tasks(story_id=story.id)
                story_data = {
                    "story": story.model_dump(mode="json"),
                    "tasks": [t.model_dump(mode="json") for t in tasks]
                }
                epic_data["stories"].append(story_data)

            result["epics"].append(epic_data)

        return result
