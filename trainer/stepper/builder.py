"""Stepper builder

Builds a stepper based on the manifest.
"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.manifest.trainer import TrainerType
from caramba.trainer.stepper.base import Stepper
from caramba.trainer.stepper.phase import PhaseStepper


class StepperBuilder:
    """Builder for creating stepper instances from manifest configuration"""

    def __init__(self, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target

    def build(self) -> Stepper:
        """Build stepper from trainer type configuration"""
        match self.target.trainer.type:
            case TrainerType.STEPWISE | TrainerType.PHASE:
                return PhaseStepper(manifest=self.manifest, target=self.target)
            case _:
                raise ValueError(
                    f"Unsupported trainer type: {self.target.trainer.type}. "
                    f"Expected one of: {[t.value for t in TrainerType]}"
                )