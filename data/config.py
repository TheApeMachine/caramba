"""Dataset configuration

Defines typed configuration objects that specify how datasets should be loaded
and prepared. Using structured configs instead of raw dictionaries makes it
easier to validate settings and catch errors before training starts.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict
import json
from pathlib import Path
from datasets import load_dataset, IterableDatasetDict, IterableDataset

from data.transforms import Transform
from data.error import DataError, DataErrorType


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
    err: DataError | None = None
