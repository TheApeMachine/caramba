"""Graph module

The graph module contains the graph for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

from caramba.manifest.node import Node

class Graph(BaseModel):
    """A graph configuration."""
    inputs: List[str] = Field(..., description="List of inputs")
    nodes: List[Node] = Field(..., description="List of nodes")