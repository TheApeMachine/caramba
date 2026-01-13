"""Logger manifest

Holds values on a logger for the manifest
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class LoggerType(str, Enum):
    """Logger type enumeration."""
    RICH = "rich"
    PLAIN = "plain"


class Logger(BaseModel):
    """A logger configuration."""
    type: LoggerType = Field(..., description="Type of logger")