"""Run context"""
from __future__ import annotations
from dataclasses import Field

from pydantic import NonNegativeInt, Field

from caramba.trainer.context.base import BaseContext
from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.manifest.target import TargetType

class RunCtx(BaseContext):
    """Run context"""
    run_id: str = Field(description="The run ID")
    phase: str = Field(description="The phase")
    step: NonNegativeInt = Field(description="The step")
    block_index: NonNegativeInt | None = Field(default=None, description="The block index")
    block_step: NonNegativeInt | None = Field(default=None, description="The block step")
    global_step: NonNegativeInt | None = Field(default=None, description="The global step")

    @classmethod
    def from_target(
        cls, *, manifest: Manifest, target: Target
    ) -> "RunCtx":
        if target.type == TargetType.EXPERIMENT:
            return cls(
                device=manifest.device,
                teacher=manifest.teacher,
                student=manifest.student,
                run_id=target.run_id,
                phase=target.phase,
                step=target.step,
            )
        else:
            raise ValueError(f"Unsupported target type: {target.type}")
