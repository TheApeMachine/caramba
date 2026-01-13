"""Dataset configuration

Defines typed configuration objects that specify how datasets should be loaded
and prepared. Using structured configs instead of raw dictionaries makes it
easier to validate settings and catch errors before training starts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator
import json
from pathlib import Path
from datasets import load_dataset, IterableDatasetDict, IterableDataset

from caramba.data.transforms import Transform
from caramba.data.error import DataError, DataErrorType


class DatasetType(Enum):
    """Dataset type

    Enumeration of supported dataset formats, allowing the framework to route
    configuration to the correct dataset builder based on the data format.
    """
    NPY = "npy"
    TOKENS = "tokens"
    TENSORS = "tensors"
    CODE_CHUNKS = "code_chunks"
    TENSOR_FILES = "tensor_files"


class DatasetConfig(BaseModel):
    """Dataset configuration

    Base configuration model that captures common dataset parameters like source
    location and token budget. Specific dataset types extend this with their
    own required fields while inheriting validation and serialization behavior.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: DatasetType
    source: str
    tokens: int
    tokenizer: str | None = None
    block_size: int | None = None
    err: DataError | None = None

    @field_validator("tokens", mode="before")
    @classmethod
    def parse_human_readable_number(cls, v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            v = v.lower().strip()
            multiplier = 1
            if v.endswith("k"):
                multiplier = 1_000
                v = v[:-1]
            elif v.endswith("m"):
                multiplier = 1_000_000
                v = v[:-1]
            elif v.endswith("b"):
                multiplier = 1_000_000_000
                v = v[:-1]
            
            try:
                return int(float(v) * multiplier)
            except ValueError:
                raise ValueError(f"Could not parse token count: {v}")
        return v
