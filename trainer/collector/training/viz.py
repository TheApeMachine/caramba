"""Visualization payload hook."""

from __future__ import annotations

from typing import Any

from caramba.collector.training import TrainHook
from caramba.instrumentation import RunLogger


class VizHook(TrainHook):
    def __init__(self, *, run_logger: RunLogger, run_id: str, phase: str) -> None:
        self._run_logger = run_logger
        self._run_id = str(run_id)
        self._phase = str(phase)

    def on_step_end(
        self,
        *,
        step: int,
        metrics: dict[str, float],
        outputs,
        batch,
        extras: dict[str, Any] | None = None,
    ) -> None:
        if not extras:
            return
        data = extras.get("viz", None)
        if not isinstance(data, dict):
            return
        self._run_logger.log_event(
            type="viz",
            run_id=self._run_id,
            phase=self._phase,
            step=int(step),
            data=data,
        )

