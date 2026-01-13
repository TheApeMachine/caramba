"""Training hook interface.

Hooks are optional, composable observers that can:
- log metrics (stdout, JSONL, W&B)
- save checkpoints
- collect layer stats / telemetry
- run periodic evaluation
"""

from __future__ import annotations

from typing import Any

from caramba.runtime.tensordict_utils import TensorDictBase


class TrainHook:
    """Composable training hook with no-op defaults."""

    def on_run_begin(self) -> None:
        return None

    def on_step_begin(self, *, step: int) -> None:
        return None

    def on_step_end(
        self,
        *,
        step: int,
        metrics: dict[str, float],
        outputs: object | None,
        batch: TensorDictBase | None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_run_end(self, *, step: int) -> None:
        return None

    def close(self) -> None:
        return None
