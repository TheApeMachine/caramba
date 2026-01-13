"""Scheduler module

The scheduler module contains the scheduler for the manifest.
"""
from __future__ import annotations

from enum import Enum
from pydantic import (
    BaseModel, Field, NonNegativeInt, NonNegativeFloat, PositiveInt, StrictBool
)


class SchedulerType(str, Enum):
    """Scheduler type enumeration."""
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"


class Scheduler(BaseModel):
    """A scheduler configuration."""
    type: SchedulerType = Field(..., description="Type of scheduler")
    warmup_steps: NonNegativeInt = Field(..., description="Warmup steps")
    min_lr_ratio: NonNegativeFloat = Field(..., description="Minimum learning rate ratio")
    auto_resume: StrictBool = Field(..., description="Whether to auto-resume")
    total_steps: PositiveInt = Field(..., description="Total steps")