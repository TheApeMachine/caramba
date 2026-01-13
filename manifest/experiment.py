"""Experiment module

The experiment module contains the experiment for the manifest.
"""
from __future__ import annotations

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

from caramba.manifest.dataset import Dataset
from caramba.manifest.trainer import Trainer
from caramba.manifest.system import System
from caramba.manifest.run import Run
from caramba.manifest.benchmark import Benchmark


class BackendType(str, Enum):
    """Backend type enumeration."""
    TORCH = "torch"
    JAX = "jax"
    TENSORFLOW = "tensorflow"
    PADDLE = "paddle"
    ONNX = "onnx"
    PLAIN = "plain"


class TaskType(str, Enum):
    """Task type enumeration."""
    LANGUAGE_MODELING = "language_modeling"
    CLASSIFICATION = "classification"
    NODE_CLASSIFICATION = "node_classification"
    DENOISING = "denoising"
    CODEGEN = "codegen"
    CCL = "ccl"
    GENERIC = "generic"


class ObjectiveType(str, Enum):
    """Objective type enumeration."""
    NEXT_TOKEN_CE = "next_token_ce"
    CROSS_ENTROPY = "cross_entropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    ROOT_MEAN_ABSOLUTE_ERROR = "root_mean_absolute_error"


class Experiment(BaseModel):
    """An experiment configuration."""
    name: str = Field(..., description="Name of the experiment")
    description: str = Field(..., description="Human-readable description")
    backend: BackendType = Field(..., description="Backend to use")
    task: TaskType = Field(..., description="Task type")
    datasets: List[Dataset] = Field(..., description="List of datasets")
    system: System = Field(..., description="System reference")
    objective: ObjectiveType = Field(..., description="Objective reference")
    trainer: Trainer = Field(..., description="Trainer reference")
    runs: List[Run] = Field(..., description="List of runs")
    benchmarks: List[Benchmark] = Field(..., description="List of benchmarks")
