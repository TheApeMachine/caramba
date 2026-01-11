"""Compute configuration: defining where and how to run.

Targets can specify a `compute` block to override global defaults.
This allows manifest-driven "rented" training runs on cloud providers
like Vast.ai.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from pydantic import BaseModel, Field

from caramba.config import PositiveFloat, PositiveInt


class LocalComputeConfig(BaseModel):
    """Configuration for local execution."""
    type: Literal["local"] = "local"
    device: str = "cpu"
    # Optional limit on local resource usage (mostly for documentation/safety).
    max_gpus: PositiveInt | None = None


class VastAIComputeConfig(BaseModel):
    """Configuration for Vast.ai execution."""
    type: Literal["vast_ai"] = "vast_ai"

    # Search criteria
    gpu_name: str | None = "RTX 4090"
    min_vram: PositiveInt = 24
    min_cuda_version: str = "12.1"

    # Budget/Safety
    max_price_per_hr: PositiveFloat = 1.0
    max_duration_hrs: PositiveFloat = 24.0
    budget_total: PositiveFloat | None = None

    # Instance lifecycle
    # If true, the instance is automatically terraformed before each run
    # and destroyed after.
    ephemeral: bool = True

    # Image/Command
    image: str = "pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel"
    # Optional path to an SSH public key for access.
    ssh_key_path: str | None = None


ComputeConfig: TypeAlias = Annotated[
    LocalComputeConfig | VastAIComputeConfig,
    Field(discriminator="type"),
]
