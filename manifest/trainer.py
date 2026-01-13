"""Trainer configuration module

The trainer configuration module contains models for trainer configurations.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat

from caramba.manifest.optimizer import Optimizer
from caramba.manifest.scheduler import Scheduler
from caramba.manifest.device import Device


class TrainerType(str, Enum):
    """Trainer type enumeration."""
    STEPWISE = "stepwise"
    BLOCKWISE = "blockwise"


class DataType(str, Enum):
    """Data type enumeration."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    UINT8 = "uint8"


class Trainer(BaseModel):
    """Trainer variables shared across targets and runs."""

    type: TrainerType = Field(..., description="Trainer type")
    steps: PositiveInt = Field(..., description="Number of training steps")
    steps_extended: PositiveInt | None = Field(
        default=None, description="Optional extended training steps"
    )
    device: Device = Field(..., description="Training device")
    dtype: DataType = Field(..., description="Data type")
    batch_size: PositiveInt = Field(..., description="Batch size")
    grad_accum: PositiveInt = Field(..., description="Gradient accumulation steps")
    lr: NonNegativeFloat = Field(..., description="Learning rate")
    lr_decoupled: NonNegativeFloat | None = Field(
        default=None, description="Optional learning rate for decoupled attention"
    )
    lr_2e4: NonNegativeFloat | None = Field(
        default=None, description="Optional sweep learning rate 2e-4"
    )
    lr_4e4: NonNegativeFloat | None = Field(
        default=None, description="Optional sweep learning rate 4e-4"
    )
    optimizer: Optimizer = Field(..., description="Optimizer")
    scheduler: Scheduler = Field(..., description="Scheduler")
    save_every: PositiveInt = Field(..., description="Save checkpoint every N steps")
