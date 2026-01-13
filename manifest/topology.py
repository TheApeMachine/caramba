"""Topology module

The topology module contains the topology for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

from caramba.manifest.layer import Layer


class TopologyType(str, Enum):
    """Topology type enumeration."""
    STACKED = "stacked"
    RESIDUAL = "residual"
    NESTED = "nested"
    PARALLEL = "parallel"
    RECURRENT = "recurrent"
    GRAPH = "graph"
    BRANCHING = "branching"
    CYCLIC = "cyclic"
    SEQUENTIAL = "sequential"


class Topology(BaseModel):
    """A topology configuration."""
    type: TopologyType = Field(..., description="Topology type")
    layers: List[Layer] = Field(..., description="List of layers")