"""Table 2 export hook (MOSAIC)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch.nn as nn

from caramba.collector.training import TrainHook
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.console import logger
from caramba.instrumentation import RunLogger
from caramba.layer.memory_block.block import MemoryBlockLayer
from caramba.trainer.mosaic_table2 import Table2SummaryWriter


class Table2ExportHook(TrainHook):
    def __init__(
        self,
        *,
        enabled: bool,
        checkpoint_dir: Path,
        target: ExperimentTargetConfig,
        run: Run,
        system: object,
        dataset_comp: object,
        table2_writer: Table2SummaryWriter,
        table2_cfg: Any,
        run_logger: RunLogger,
    ) -> None:
        self._enabled = bool(enabled)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._target = target
        self._run = run
        self._system = system
        self._dataset_comp = dataset_comp
        self._writer = table2_writer
        self._cfg = table2_cfg
        self._run_logger = run_logger
        self._last_metrics: dict[str, float] | None = None

    def on_step_end(self, *, step: int, metrics: dict[str, float], outputs, batch, extras=None) -> None:
        if not self._enabled:
            return
        if not metrics:
            return
        if any(str(k).startswith("acc/bin_") for k in metrics.keys()):
            self._last_metrics = dict(metrics)

    def on_run_end(self, *, step: int) -> None:
        if not self._enabled:
            return
        if not bool(getattr(self._run_logger, "enabled", True)):
            return
        if self._last_metrics is None:
            return
        if not any(k.startswith("acc/bin_") for k in self._last_metrics.keys()):
            return
        self._export_table2_bundle(last_table2_metrics=self._last_metrics)

    def _export_table2_bundle(self, *, last_table2_metrics: dict[str, float]) -> None:
        # Prefer extracting memory config from the model (for MOSAIC). Fall back to dataset component when needed.
        mb: int | None = None
        mh: int | None = None
        mod = getattr(self._system, "module", None)
        if isinstance(mod, nn.Module):
            buckets: set[int] = set()
            hashes: set[int] = set()
            for m in mod.modules():
                if isinstance(m, MemoryBlockLayer):
                    buckets.add(int(m.memory.mem_buckets))
                    hashes.add(int(m.memory.mem_hashes))
            if buckets and hashes:
                if len(buckets) != 1 or len(hashes) != 1:
                    raise ValueError("Inconsistent mem_buckets/mem_hashes across MemoryBlockLayer modules.")
                mb = next(iter(buckets))
                mh = next(iter(hashes))

        if mb is None or mh is None:
            mb2 = getattr(self._dataset_comp, "mem_buckets", None)
            mh2 = getattr(self._dataset_comp, "mem_hashes", None)
            if isinstance(mb2, int) and isinstance(mh2, int):
                mb = int(mb2)
                mh = int(mh2)

        if mb is None or mh is None:
            raise TypeError("Table 2 export requires mem_buckets/mem_hashes (from model or dataset).")

        params_fn = getattr(self._system, "parameters", None)
        if not callable(params_fn):
            raise TypeError("Table 2 export requires system.parameters().")

        n_params = 0
        params_iter = params_fn()
        if not isinstance(params_iter, Iterable):
            raise TypeError("system.parameters() must return an iterable")
        for p in params_iter:
            if not isinstance(p, nn.Parameter):
                raise TypeError("system.parameters() must yield nn.Parameter objects")
            n_params += int(p.numel())

        out_path = self._writer.write(
            out_dir=self._checkpoint_dir,
            run_id=str(self._run.id),
            mem_buckets=int(mb),
            mem_hashes=int(mh),
            model_size=f"params={n_params}",
            metrics=last_table2_metrics,
            n_bins=int(self._cfg.n_bins),
        )

        # Console UX: print both the path and a small table summary.
        logger.subheader("Table 2 export")
        logger.path(str(out_path), label="summary_json")
        logger.info(f"Table 2 summary written to [path]{out_path}[/path]")

        rows: list[list[str]] = []
        rows.append(["mem_buckets", str(int(mb))])
        rows.append(["mem_hashes", str(int(mh))])
        rows.append(["model_size", f"params={n_params}"])

        wb = float(last_table2_metrics.get("acc/worst_bin", -1.0))
        cr = float(last_table2_metrics.get("collision/wrong_item_read_rate", -1.0))
        rows.append(["acc/worst_bin", "—" if wb < 0.0 else f"{wb:.4f}"])
        rows.append(["collision/wrong_item_read_rate", "—" if cr < 0.0 else f"{cr:.4f}"])

        for i in range(int(self._cfg.n_bins)):
            k = f"acc/bin_{i}"
            v = float(last_table2_metrics.get(k, -1.0))
            rows.append([k, "—" if v < 0.0 else f"{v:.4f}"])

        logger.table(
            title=f"Table 2 summary • {self._target.name}:{self._run.id}",
            columns=["Metric", "Value"],
            rows=rows,
        )

