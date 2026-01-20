"""Instrumentation configuration with discriminated unions.

Follows the same pattern as layer configuration: each backend type has its
own config class, and Config.build() instantiates the correct implementation.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from config import Config


class InstrumentationType(str, enum.Enum):
    """Available instrumentation backend types."""
    HDF5 = "H5Store"
    TB = "TensorBoardWriter"
    WANDB = "WandBWriter"
    LIVEPLOT = "LivePlotter"
    JSONL = "RunLogger"

    @staticmethod
    def module_name() -> str:
        return "caramba.instrumentation"

    def py_module(self) -> str:
        # Map config enum members to implementation filenames.
        # (Enum member names are schema-level; filenames are implementation-level.)
        match self:
            case InstrumentationType.HDF5:
                return "hdf5_store"
            case InstrumentationType.TB:
                return "tensorboard_writer"
            case InstrumentationType.WANDB:
                return "wandb_writer"
            case InstrumentationType.LIVEPLOT:
                return "live_plotter"
            case InstrumentationType.JSONL:
                return "run_logger"
            case _:
                return self.name.lower()


class HDF5Config(Config):
    """HDF5 array storage."""
    type: Literal[InstrumentationType.HDF5] = InstrumentationType.HDF5
    path: str = "data.h5"
    enabled: bool = True


class TensorBoardConfig(Config):
    """TensorBoard logging."""
    type: Literal[InstrumentationType.TB] = InstrumentationType.TB
    out_dir: str = "tb"
    enabled: bool = True
    log_every: int = 10


class WandBConfig(Config):
    """Weights & Biases logging."""
    type: Literal[InstrumentationType.WANDB] = InstrumentationType.WANDB
    out_dir: str = "wandb"
    enabled: bool = True
    project: str = ""
    entity: str = ""
    mode: str = "online"
    run_name: str = ""
    group: str = ""
    tags: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)


class LivePlotConfig(Config):
    """Live matplotlib plotting."""
    type: Literal[InstrumentationType.LIVEPLOT] = InstrumentationType.LIVEPLOT
    enabled: bool = True
    title: str = "training"
    plot_every: int = 10


class JSONLConfig(Config):
    """JSONL event logging."""
    type: Literal[InstrumentationType.JSONL] = InstrumentationType.JSONL
    out_dir: str = "."
    filename: str = "train.jsonl"
    enabled: bool = True


InstrumentationConfig: TypeAlias = Annotated[
    HDF5Config | TensorBoardConfig | WandBConfig | LivePlotConfig | JSONLConfig,
    Field(discriminator="type"),
]
