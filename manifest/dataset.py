"""Dataset manifest

Holds values on a dataset for the manifest
"""
from __future__ import annotations
from typing import Union
from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat


class Dataset(BaseModel):
    """A dataset configuration."""
    repo: str = Field(..., description="Repository identifier")
    tokens: Union[str, PositiveInt] = Field(..., description="Number of tokens")
    tokenizer: str = Field(..., description="Tokenizer identifier")
    block_size: PositiveInt = Field(..., description="Block size for sequences")
    value_fraction: NonNegativeFloat = Field(..., description="Value fraction")