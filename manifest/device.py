"""Device manifest

Holds values on a device for the manifest
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class DeviceType(str, Enum):
    """Device type enumeration."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Device(BaseModel):
    """A device configuration."""
    type: DeviceType = Field(..., description="Device type")

    @model_validator(mode="before")
    @classmethod
    def coerce_str(cls, v: object) -> object:
        """Allow representing a device as either a scalar or an object.

        YAML often uses a scalar form:

        ```yaml
        device: mps
        ```
        """
        if isinstance(v, str):
            return {"type": v}
        return v
