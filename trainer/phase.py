"""Phase-based trainer

Trains models using phase-based progression, coordinating between different
training phases (blockwise, global, orchestrated) as defined in the manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.base import Trainer
from caramba.trainer.steppers.builder import StepperBuilder


class PhaseTrainer(Trainer):
    def __init__(
        self,
        *,
        manifest: Manifest,
        target: Target,
    ) -> None:
        super().__init__(manifest=manifest, target=target)
        self.manifest = manifest
        self.target = target
        self.stepper = StepperBuilder(manifest=manifest, target=target).build()

    def run(self) -> dict[str, Any] | None:
        while self.stepper.step():
            pass
