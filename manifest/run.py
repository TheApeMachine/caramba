"""Run module

The run module contains the run for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt


class RunMode(str, Enum):
    """Run mode enumeration."""
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"


class Run(BaseModel):
    """A run configuration within an experiment."""
    id: str = Field(..., description="Unique identifier for the run")
    mode: RunMode = Field(..., description="Mode of the run")
    seed: NonNegativeInt = Field(..., description="Random seed")
    steps: PositiveInt = Field(..., description="Number of steps to run")
