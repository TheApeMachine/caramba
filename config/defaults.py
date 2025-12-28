"""Default settings that apply across all runs in a manifest.

Core philosophy alignment:
- Manifests should express intent at the right level of abstraction.
- Defaults are grouped by concern (data/logging/runtime) to reduce noise and
  make it clearer which knobs matter for which layer of the system.
"""
from __future__ import annotations

from pydantic import BaseModel

from config import PositiveInt, Probability


class DefaultsData(BaseModel):
    """Defaults for dataset/tokenizer related behavior."""

    tokenizer: str = "tiktoken"
    val_frac: Probability = 0.1


class DefaultsLogging(BaseModel):
    """Defaults for logging/instrumentation backends."""

    instrument: str = "rich"
    wandb: bool = True
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_mode: str = "online"
    eval_iters: PositiveInt = 50


class DefaultsRuntime(BaseModel):
    """Defaults for runtime/execution behaviors."""

    save_every: PositiveInt = 100
    # If true, allow "best effort" fallbacks for optional perf backends (Metal/Triton),
    # but log high-visibility warnings when acceleration is unavailable.
    best_effort: bool = False


class Defaults(BaseModel):
    """Global defaults shared across all runs in a manifest."""

    data: DefaultsData = DefaultsData()
    logging: DefaultsLogging = DefaultsLogging()
    runtime: DefaultsRuntime = DefaultsRuntime()
