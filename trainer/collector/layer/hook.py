"""TrainHook implementation for layer stats."""

from __future__ import annotations

from caramba.collector.layer.manager import LayerStatsManager
from caramba.collector.training import TrainHook
from caramba.instrumentation import RunLogger


class LayerStatsHook(TrainHook):
    def __init__(
        self,
        *,
        system: object,
        interval: int,
        run_logger: RunLogger,
        run_id: str,
        phase: str,
    ) -> None:
        self._mgr = LayerStatsManager(system=system, interval=int(interval))
        self._run_logger = run_logger
        self._run_id = str(run_id)
        self._phase = str(phase)

    def on_step_begin(self, *, step: int) -> None:
        self._mgr.begin_step(int(step))

    def on_step_end(self, *, step: int, metrics: dict[str, float], outputs, batch, extras=None) -> None:
        self._mgr.end_step()
        if not self._mgr.should_log(int(step)):
            return
        self._run_logger.log_event(
            type="layer_stats",
            run_id=self._run_id,
            phase=self._phase,
            step=int(step),
            data=self._mgr.payload(),
        )

    def close(self) -> None:
        self._mgr.close()

