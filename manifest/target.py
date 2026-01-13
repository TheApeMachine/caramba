"""Target module

The target module contains the target for the manifest.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field


class TargetType(str, Enum):
    """Target type enumeration."""
    EXPERIMENT = "experiment"
    PROCESS = "process"


class Target(BaseModel):
    """The target for the manifest."""
    type: TargetType = Field(..., description="Type of target")
    name: str = Field(..., description="Unique name of the target")
    description: str = Field(..., description="Human-readable description")
    backend: Literal["torch"] = Field(..., description="Backend to use")
    task: Literal["task.language_modeling"] = Field(..., description="Task type")
    data: Dict[str, Any] = Field(..., description="Data configuration")
    system: Dict[str, Any] = Field(..., description="System configuration")
    objective: Literal["objective.next_token_ce"] = Field(..., description="Objective function")
    trainer: Literal["trainer.train"] = Field(..., description="Trainer configuration")
    runs: List[Dict[str, Any]] = Field(..., description="List of run configurations")
    benchmarks: Optional[List[Dict[str, Any]]] = Field(None, description="Optional benchmark configurations")