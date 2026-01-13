"""Instrumentation manifest

Holds values on a instrumentation for the manifest
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from caramba.manifest.logger import Logger
from caramba.manifest.metrics import Metrics


class Instrumentation(BaseModel):
    """An instrumentation configuration."""
    logger: Logger = Field(..., description="Logger")
    metrics: Metrics = Field(..., description="Metrics")