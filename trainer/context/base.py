"""Base context class"""
from __future__ import annotations
from abc import abstractmethod

from pydantic import BaseModel, Field
import torch
from torch import nn


class BaseContext(BaseModel):
    """Base context class"""
    device: torch.device = Field(description="The device to use")
    teacher: nn.Module = Field(description="The teacher model")
    student: nn.Module = Field(description="The student model")