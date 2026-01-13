"""Logging hooks for training."""

from __future__ import annotations

from caramba.collector.training import TrainHook
from caramba.console import logger
from caramba.instrumentation import RunLogger
from caramba.instrumentation.wandb_writer import WandBWriter


class RunLoggerHook(TrainHook):
    def __init__(self, *, run_logger: RunLogger, run_id: str, phase: str) -> None:
        self._run_logger = run_logger
        self._run_id = str(run_id)
        self._phase = str(phase)

    def on_step_end(self, *, step: int, metrics: dict[str, float], outputs, batch, extras=None) -> None:
        if not metrics:
            return
        self._run_logger.log_metrics(
            run_id=self._run_id,
            phase=self._phase,
            step=int(step),
            metrics=metrics,
        )


class WandBHook(TrainHook):
    def __init__(self, *, writer: WandBWriter, prefixes: tuple[str, ...] = ("train", "")) -> None:
        self._writer = writer
        self._prefixes = tuple(str(p) for p in prefixes)

    def on_step_end(self, *, step: int, metrics: dict[str, float], outputs, batch, extras=None) -> None:
        if not metrics:
            return
        for prefix in self._prefixes:
            try:
                self._writer.log_scalars(prefix=str(prefix), step=int(step), scalars=metrics)
            except Exception as e:
                logger.fallback_warning(
                    "WARNING: W&B logging failed for this step (continuing).\n"
                    f"reason={type(e).__name__}: {e}"
                )

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception as e:
            logger.warning(f"Failed to close W&B writer (ignoring). error={e!r}")

