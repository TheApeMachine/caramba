"""Bootstrap context passed to Initializer components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from caramba.config.defaults import Defaults
from caramba.config.group import Group
from caramba.runtime import RuntimePlan
from caramba.trainer.distributed import DistributedContext


@dataclass(frozen=True, slots=True)
class UpcycleInitContext:
    # Kept intentionally loose: upcycling can be driven by different manifest schemas.
    manifest: object
    group: Group
    defaults: Defaults | None

    checkpoint_dir: Path
    device: torch.device
    dtype: torch.dtype
    runtime_plan: RuntimePlan

    dist_ctx: DistributedContext | None

