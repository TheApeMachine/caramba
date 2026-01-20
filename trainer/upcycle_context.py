"""Shared context object passed from Upcycle (orchestrator) to pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from config.defaults import Defaults
from config.group import Group
from runtime import RuntimePlan
from instrumentation import Instrumentation
from trainer.distributed import DistributedContext


@dataclass(frozen=True, slots=True)
class UpcycleContext:
    # Kept intentionally loose: upcycling can be driven by different manifest schemas.
    manifest: object
    group: Group
    defaults: Defaults | None

    checkpoint_dir: Path
    device: torch.device
    dtype: torch.dtype
    runtime_plan: RuntimePlan

    teacher: nn.Module
    student: nn.Module

    inst: Instrumentation | None
    dist_ctx: DistributedContext | None

