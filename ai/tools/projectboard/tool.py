"""Project Board MCP Server.

Provides project management capabilities with hierarchical structure:
Projects > Epics > Stories > Tasks

This tool enables AI agents to:
- Create and manage projects, epics, stories, and tasks
- Track dependencies between work items
- Query and filter work items by status, priority, etc.
- Get full project hierarchies
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Optional
from uuid import UUID

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .models import Epic, Priority, Project, Status, Story, Task
from .storage import ProjectBoardStorage

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ProjectBoard Tools", json_response=True)
mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

# Global storage instance
storage: Optional[ProjectBoardStorage] = None


async def get_storage() -> ProjectBoardStorage:
    """Get or create the storage instance."""
    global storage
    if storage is None:
        database_url = os.environ.get(
            "PROJECTBOARD_DATABASE_URL",
            "postgresql://projectboard:projectboard@postgres-projectboard:5432/projectboard"
        )
        storage = await ProjectBoardStorage.create(database_url)
    return storage


@mcp.tool()
async def create_project(
    title: str,
    description: str = "",
    objectives: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Create a new project.

    Args:
        title: Project title
        description: Detailed description of the project
        objectives: List of project objectives/goals
        tags: Tags for categorization

    Returns:
        Created project data
    """
    store = await get_storage()
    project = Project(
        title=title,
        description=description,
        objectives=objectives or [],
        tags=tags or [],
    )
    await store.create_project(project)
    return project.model_dump(mode="json")


@mcp.tool()
async def list_projects(status: str | None = None) -> list[dict]:
    """List all projects, optionally filtered by status.

    Args:
        status: Filter by status (todo, in_progress, done, blocked, cancelled)

    Returns:
        List of projects
    """
    store = await get_storage()
    status_filter = Status(status) if status else None
    projects = await store.list_projects(status=status_filter)
    return [p.model_dump(mode="json") for p in projects]


@mcp.tool()
async def update_project_status(project_id: str, status: str) -> dict:
    """Update a project's status.

    Args:
        project_id: UUID of the project
        status: New status (todo, in_progress, done, blocked, cancelled)

    Returns:
        Updated project data
    """
    store = await get_storage()
    project = await store.get_project(UUID(project_id))
    if not project:
        return {"error": f"Project {project_id} not found"}

    project.status = Status(status)
    if status == "done":
        from datetime import datetime
        project.completed_at = datetime.utcnow()

    await store.update_project(project)
    return project.model_dump(mode="json")


@mcp.tool()
async def create_epic(
    project_id: str,
    title: str,
    description: str = "",
    goals: list[str] | None = None,
    priority: str = "medium",
    tags: list[str] | None = None,
) -> dict:
    """Create a new epic within a project.

    Args:
        project_id: UUID of the parent project
        title: Epic title
        description: Detailed description
        goals: List of epic goals
        priority: Priority level (low, medium, high, critical)
        tags: Tags for categorization

    Returns:
        Created epic data
    """
    store = await get_storage()
    epic = Epic(
        project_id=UUID(project_id),
        title=title,
        description=description,
        goals=goals or [],
        priority=Priority(priority),
        tags=tags or [],
    )
    await store.create_epic(epic)
    return epic.model_dump(mode="json")


@mcp.tool()
async def list_epics(project_id: str | None = None, status: str | None = None) -> list[dict]:
    """List epics, optionally filtered by project and/or status.

    Args:
        project_id: Filter by project UUID
        status: Filter by status (todo, in_progress, done, blocked, cancelled)

    Returns:
        List of epics
    """
    store = await get_storage()
    proj_filter = UUID(project_id) if project_id else None
    status_filter = Status(status) if status else None
    epics = await store.list_epics(project_id=proj_filter, status=status_filter)
    return [e.model_dump(mode="json") for e in epics]


@mcp.tool()
async def update_epic_status(epic_id: str, status: str) -> dict:
    """Update an epic's status.

    Args:
        epic_id: UUID of the epic
        status: New status (todo, in_progress, done, blocked, cancelled)

    Returns:
        Updated epic data
    """
    store = await get_storage()
    epics = await store.list_epics()
    epic = next((e for e in epics if str(e.id) == epic_id), None)
    if not epic:
        return {"error": f"Epic {epic_id} not found"}

    epic.status = Status(status)
    if status == "done":
        from datetime import datetime
        epic.completed_at = datetime.utcnow()

    await store.update_epic(epic)
    return epic.model_dump(mode="json")


@mcp.tool()
async def create_story(
    epic_id: str,
    title: str,
    description: str = "",
    acceptance_criteria: list[str] | None = None,
    priority: str = "medium",
    assignee: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Create a new story within an epic.

    Args:
        epic_id: UUID of the parent epic
        title: Story title
        description: Detailed description
        acceptance_criteria: List of acceptance criteria
        priority: Priority level (low, medium, high, critical)
        assignee: Assigned agent/person name
        tags: Tags for categorization

    Returns:
        Created story data
    """
    store = await get_storage()
    story = Story(
        epic_id=UUID(epic_id),
        title=title,
        description=description,
        acceptance_criteria=acceptance_criteria or [],
        priority=Priority(priority),
        assignee=assignee,
        tags=tags or [],
    )
    await store.create_story(story)
    return story.model_dump(mode="json")


@mcp.tool()
async def list_stories(epic_id: str | None = None, status: str | None = None) -> list[dict]:
    """List stories, optionally filtered by epic and/or status.

    Args:
        epic_id: Filter by epic UUID
        status: Filter by status (todo, in_progress, done, blocked, cancelled)

    Returns:
        List of stories
    """
    store = await get_storage()
    epic_filter = UUID(epic_id) if epic_id else None
    status_filter = Status(status) if status else None
    stories = await store.list_stories(epic_id=epic_filter, status=status_filter)
    return [s.model_dump(mode="json") for s in stories]


@mcp.tool()
async def update_story_status(story_id: str, status: str, assignee: str | None = None) -> dict:
    """Update a story's status and optionally assignee.

    Args:
        story_id: UUID of the story
        status: New status (todo, in_progress, done, blocked, cancelled)
        assignee: Optionally update assignee

    Returns:
        Updated story data
    """
    store = await get_storage()
    stories = await store.list_stories()
    story = next((s for s in stories if str(s.id) == story_id), None)
    if not story:
        return {"error": f"Story {story_id} not found"}

    story.status = Status(status)
    if assignee:
        story.assignee = assignee
    if status == "done":
        from datetime import datetime
        story.completed_at = datetime.utcnow()

    await store.update_story(story)
    return story.model_dump(mode="json")


@mcp.tool()
async def create_task(
    story_id: str,
    title: str,
    description: str = "",
    priority: str = "medium",
    assignee: str | None = None,
    depends_on: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Create a new task within a story.

    Args:
        story_id: UUID of the parent story
        title: Task title
        description: Detailed description
        priority: Priority level (low, medium, high, critical)
        assignee: Assigned agent/person name
        depends_on: List of task UUIDs this task depends on
        tags: Tags for categorization

    Returns:
        Created task data
    """
    store = await get_storage()
    task = Task(
        story_id=UUID(story_id),
        title=title,
        description=description,
        priority=Priority(priority),
        assignee=assignee,
        depends_on=[UUID(d) for d in (depends_on or [])],
        tags=tags or [],
    )
    await store.create_task(task)
    return task.model_dump(mode="json")


@mcp.tool()
async def list_tasks(story_id: str | None = None, status: str | None = None) -> list[dict]:
    """List tasks, optionally filtered by story and/or status.

    Args:
        story_id: Filter by story UUID
        status: Filter by status (todo, in_progress, done, blocked, cancelled)

    Returns:
        List of tasks
    """
    store = await get_storage()
    story_filter = UUID(story_id) if story_id else None
    status_filter = Status(status) if status else None
    tasks = await store.list_tasks(story_id=story_filter, status=status_filter)
    return [t.model_dump(mode="json") for t in tasks]


@mcp.tool()
async def update_task_status(task_id: str, status: str, assignee: str | None = None) -> dict:
    """Update a task's status and optionally assignee.

    Args:
        task_id: UUID of the task
        status: New status (todo, in_progress, done, blocked, cancelled)
        assignee: Optionally update assignee

    Returns:
        Updated task data
    """
    store = await get_storage()
    tasks = await store.list_tasks()
    task = next((t for t in tasks if str(t.id) == task_id), None)
    if not task:
        return {"error": f"Task {task_id} not found"}

    task.status = Status(status)
    if assignee:
        task.assignee = assignee
    if status == "done":
        from datetime import datetime
        task.completed_at = datetime.utcnow()

    await store.update_task(task)
    return task.model_dump(mode="json")


@mcp.tool()
async def get_project_hierarchy(project_id: str) -> dict:
    """Get the full hierarchy of a project (epics > stories > tasks).

    Args:
        project_id: UUID of the project

    Returns:
        Complete project hierarchy with all nested items
    """
    store = await get_storage()
    return await store.get_project_hierarchy(UUID(project_id))


@mcp.tool()
async def search_work_items(
    query: str,
    item_types: list[str] | None = None,
    status: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Search for work items by title/description.

    Args:
        query: Search text (matches title and description)
        item_types: Filter by types: ["project", "epic", "story", "task"]
        status: Filter by status
        priority: Filter by priority
        tags: Filter by tags (items must have ALL specified tags)

    Returns:
        Matching work items grouped by type
    """
    store = await get_storage()
    result = {
        "projects": [],
        "epics": [],
        "stories": [],
        "tasks": [],
    }

    types = item_types or ["project", "epic", "story", "task"]
    query_lower = query.lower()

    if "project" in types:
        projects = await store.list_projects()
        result["projects"] = [
            p.model_dump(mode="json")
            for p in projects
            if query_lower in p.title.lower() or query_lower in p.description.lower()
        ]

    if "epic" in types:
        epics = await store.list_epics()
        result["epics"] = [
            e.model_dump(mode="json")
            for e in epics
            if query_lower in e.title.lower() or query_lower in e.description.lower()
        ]

    if "story" in types:
        stories = await store.list_stories()
        result["stories"] = [
            s.model_dump(mode="json")
            for s in stories
            if query_lower in s.title.lower() or query_lower in s.description.lower()
        ]

    if "task" in types:
        tasks = await store.list_tasks()
        result["tasks"] = [
            t.model_dump(mode="json")
            for t in tasks
            if query_lower in t.title.lower() or query_lower in t.description.lower()
        ]

    return result


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="ProjectBoard MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--database-url", help="PostgreSQL connection URL")
    args = parser.parse_args()

    if args.database_url:
        os.environ["PROJECTBOARD_DATABASE_URL"] = args.database_url

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting ProjectBoard MCP server on {args.host}:{args.port}")
    mcp.settings.host = str(args.host)
    mcp.settings.port = int(args.port)

    app = mcp.streamable_http_app()

    def root(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    app.add_route("/", root, methods=["GET"])
    app.add_route("/health", health, methods=["GET"])

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
