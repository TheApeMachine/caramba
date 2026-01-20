"""Instrumentation: structured logging for training and inference.

Backends are instantiated via Config.build() just like layers—add a config
to your manifest and the correct backend is created automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config.instrumentation import InstrumentationConfig
from instrumentation.hdf5_store import H5Store
from instrumentation.analysis import generate_analysis_png
from instrumentation.live_plotter import LivePlotter
from instrumentation.run_logger import RunLogger
from instrumentation.viz import TrainingVizContext
from instrumentation.tensorboard_writer import TensorBoardWriter
from instrumentation.wandb_writer import WandBWriter

__all__ = [
    "Instrumentation",
    "H5Store",
    "LivePlotter",
    "RunLogger",
    "TrainingVizContext",
    "TensorBoardWriter",
    "WandBWriter",
    "generate_analysis_png",
]


class Instrumentation:
    """Unified facade that dispatches to configured backends."""

    def __init__(self, configs: list[InstrumentationConfig], out_dir: Path) -> None:
        self.backends: list[Any] = [cfg.build() for cfg in configs]
        self.out_dir = out_dir

    def log_scalars(self, *, step: int, prefix: str, scalars: dict[str, float]) -> None:
        for b in self.backends:
            fn = getattr(b, "log_scalars", None)
            if callable(fn):
                fn(prefix=prefix, step=step, scalars=scalars)

    def log_histogram(self, *, step: int, name: str, values: Any) -> None:
        for b in self.backends:
            fn = getattr(b, "log_histogram", None)
            if callable(fn):
                fn(name=name, step=step, values=values)

    def close(self) -> None:
        for b in self.backends:
            fn = getattr(b, "close", None)
            if callable(fn):
                fn()

    def __enter__(self) -> "Instrumentation":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
