"""Bootstrap context passed to Initializer components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from config.defaults import Defaults
from config.group import Group
from config.manifest import Manifest
from runtime import RuntimePlan
from trainer.distributed import DistributedContext


@dataclass(frozen=True, slots=True)
class UpcycleInitContext:
    manifest: Manifest
    group: Group
    defaults: Defaults | None

    checkpoint_dir: Path
    device: torch.device
    dtype: torch.dtype
    runtime_plan: RuntimePlan

    dist_ctx: DistributedContext | None

