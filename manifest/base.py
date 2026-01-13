"""Manifest base class

The manifest is the single source of truth that governs the entire
caramba research substrate.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, PositiveInt

from caramba.manifest.intrumentation import Instrumentation
from caramba.manifest.variables import Variables
from caramba.manifest.entrypoints import Entrypoints
from caramba.manifest.target import Target


class BaseManifest(BaseModel):
    """The main manifest configuration."""
    version: PositiveInt = Field(..., description="Manifest version")
    name: str = Field(..., description="Manifest name")
    notes: str = Field("", description="Human-readable notes about the manifest")
    instrumentation: Instrumentation | None = Field(
        default=None, description="Optional instrumentation configuration"
    )
    variables: Variables = Field(..., description="Variables used throughout the manifest")
    targets: List[Target] = Field(..., description="List of target configurations")
    entrypoints: Entrypoints | None = Field(
        default=None, description="Optional entrypoints for easy access"
    )
