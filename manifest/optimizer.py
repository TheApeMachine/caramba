"""Optimizer module

The optimizer module contains the optimizer for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field, NonNegativeFloat, StrictBool


class OptimizerType(str, Enum):
    """Optimizer type enumeration."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LION = "lion"


class Optimizer(BaseModel):
    """An optimizer configuration."""
    type: OptimizerType = Field(..., description="Type of optimizer")
    betas: Tuple[NonNegativeFloat, NonNegativeFloat] = Field(..., description="Beta1 and Beta2")
    eps: NonNegativeFloat = Field(..., description="Epsilon")
    weight_decay: NonNegativeFloat = Field(..., description="Weight decay")
    fused: StrictBool = Field(..., description="Whether to use fused optimizer")
