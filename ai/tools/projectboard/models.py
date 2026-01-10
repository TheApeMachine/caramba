"""Data models for the project management system.

Hierarchy: Project > Epic > Story > Task
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Status of a work item."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority of a work item."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """A task is the smallest unit of work."""
    id: UUID = Field(default_factory=uuid4)
    story_id: UUID
    title: str
    description: str = ""
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    assignee: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    depends_on: list[UUID] = Field(default_factory=list)  # Task IDs this depends on
    tags: list[str] = Field(default_factory=list)


class Story(BaseModel):
    """A user story contains multiple tasks."""
    id: UUID = Field(default_factory=uuid4)
    epic_id: UUID
    title: str
    description: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    assignee: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    depends_on: list[UUID] = Field(default_factory=list)  # Story IDs this depends on
    tags: list[str] = Field(default_factory=list)


class Epic(BaseModel):
    """An epic contains multiple stories."""
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    title: str
    description: str = ""
    goals: list[str] = Field(default_factory=list)
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    depends_on: list[UUID] = Field(default_factory=list)  # Epic IDs this depends on
    tags: list[str] = Field(default_factory=list)


class Project(BaseModel):
    """A project is the top-level organizational unit."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str = ""
    objectives: list[str] = Field(default_factory=list)
    status: Status = Status.TODO
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


class DependencyGraph(BaseModel):
    """Represents dependencies between work items."""
    item_id: UUID
    item_type: str  # "task", "story", "epic"
    depends_on: list[UUID]
    blocks: list[UUID] = Field(default_factory=list)  # Items blocked by this one


class ParsedWorkStructure(BaseModel):
    """Output schema for parsing unstructured text into work items.

    This is used by the project_manager agent to convert natural language
    into structured project data.
    """
    projects: list[Project] = Field(default_factory=list)
    epics: list[Epic] = Field(default_factory=list)
    stories: list[Story] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    dependencies: list[DependencyGraph] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Explanation of how the text was parsed into the structure"
    )
