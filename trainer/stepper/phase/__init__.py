"""Phase-based training stepper

This package contains the phase-based fine-tuning training loop split into
small, composable collaborators. The public entrypoint is `PhaseStepper`.
"""

from __future__ import annotations

from caramba.trainer.stepper.phase.stepper import PhaseStepper

__all__ = [
    "PhaseStepper",
]

