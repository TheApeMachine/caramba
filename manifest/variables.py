"""Variables module

The variables module contains the variables for the manifest.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, PositiveInt

from caramba.manifest.dataset import Dataset
from caramba.manifest.model import Model
from caramba.manifest.trainer import Trainer


class Variables(BaseModel):
    """The template-time variables for a manifest.

    Variables are resolved into concrete target configs during compilation.
    """

    datasets: List[Dataset] = Field(..., description="List of datasets")
    model: Model = Field(..., description="Model configuration")

    # DBA bottleneck dimensions
    sem_dim: PositiveInt = Field(..., description="Semantic dimension for DBA")
    geo_dim: PositiveInt = Field(..., description="Geometric dimension for DBA")
    attn_dim: PositiveInt = Field(..., description="Attention dimension for DBA")

    trainer: Trainer = Field(..., description="Training defaults and sweep variables")
