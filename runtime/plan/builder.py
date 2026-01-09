"""Runtime plan builder for caching device/runtime decisions.

Trainers often need to resolve runtime knobs that depend on the environment
(dtype, AMP dtype, batch size heuristics, torch.compile mode). Those decisions
are stable for a given (device, model/system config, train config) signature and
are worth caching to disk under the run checkpoint directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from caramba.carmath import (
    autocast_dtype_str,
    token_budget_batch_size,
    weight_dtype_str,
)
from caramba.config.train import TrainConfig
from caramba.runtime.plan import RuntimePlan, load_plan, make_plan_key, save_plan


class RuntimePlanBuilder:
    """Build and cache a `RuntimePlan` under `<checkpoint_dir>/plans/<key>.json`."""

    def __init__(self, *, plans_dirname: str = "plans") -> None:
        self.plans_dirname = str(plans_dirname)

    def build(
        self,
        *,
        checkpoint_dir: Path,
        device: torch.device,
        train: TrainConfig,
        payload: dict[str, Any],
    ) -> RuntimePlan:
        key = make_plan_key(payload)
        plan_path = checkpoint_dir / self.plans_dirname / f"{key}.json"

        existing = load_plan(plan_path)
        if existing is not None and existing.key == key:
            return existing

        dtype_str = str(train.dtype).lower()
        if dtype_str == "auto":
            dtype_str = weight_dtype_str(device)

        amp_dtype_str = str(train.amp_dtype).lower()
        if amp_dtype_str == "auto":
            amp_dtype_str = autocast_dtype_str(device)

        batch_size = int(train.batch_size)
        if bool(getattr(train, "auto_batch_size", False)):
            ref_block = int(getattr(train, "auto_batch_ref_block_size", 512))
            min_bs = int(getattr(train, "auto_batch_min", 1))
            batch_size = token_budget_batch_size(
                batch_size,
                block_size=int(train.block_size),
                ref_block_size=int(ref_block),
                min_batch_size=int(min_bs),
            )

        plan = RuntimePlan(
            key=key,
            device=str(device),
            torch_version=str(getattr(torch, "__version__", "")),
            dtype=dtype_str,
            use_amp=bool(train.use_amp),
            amp_dtype=amp_dtype_str,
            batch_size=int(batch_size),
            compile=bool(getattr(train, "compile_model", False)),
            compile_mode=str(getattr(train, "compile_mode", "reduce-overhead")),
        )
        try:
            save_plan(plan_path, plan, payload=payload)
        except Exception as e:
            raise RuntimeError("Failed to save runtime plan") from e

        return plan

