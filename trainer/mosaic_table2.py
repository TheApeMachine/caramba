"""Telemetry helpers for MOSAIC Table 2 experiments.

This module implements the "production readiness checklist" metrics required by
meeting_notes_20260103_051953.md:
- distance-binned recall accuracy
- tail metric (worst bin)
- collision proxy: wrong-item read rate

These metrics are computed from (batch, logits) without requiring MOSAIC-specific
internals, so they can be used for baseline comparisons (e.g., SSMLayer).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor

from caramba.runtime.tensordict_utils import TensorDictBase


@dataclass(frozen=True, slots=True)
class Table2TelemetryConfig:
    n_bins: int = 8


@dataclass(frozen=True, slots=True)
class Table2Telemetry:
    """Compute Table 2 metrics for synthetic memory curricula."""

    cfg: Table2TelemetryConfig = Table2TelemetryConfig()
    logits_key: str = "logits"
    target_key: str = "target_ids"

    def compute(self, *, outputs: object, batch: TensorDictBase) -> dict[str, float]:
        if not isinstance(batch, TensorDictBase):
            raise TypeError(f"batch must be a TensorDictBase, got {type(batch).__name__}")

        if not isinstance(outputs, dict):
            raise TypeError(f"outputs must be a dict, got {type(outputs).__name__}")

        logits_obj = outputs.get(str(self.logits_key))
        if not isinstance(logits_obj, Tensor):
            raise KeyError(f"Missing Tensor logits under key {self.logits_key!r} in outputs")

        try:
            target_obj = batch[str(self.target_key)]
        except KeyError as e:
            raise KeyError(f"Missing Tensor targets under key {self.target_key!r} in batch") from e
        if not isinstance(target_obj, Tensor):
            raise TypeError(f"batch[{self.target_key!r}] must be a Tensor, got {type(target_obj).__name__}")

        logits = logits_obj.detach()
        target = target_obj.detach()

        if logits.ndim != 3:
            raise ValueError(f"Expected logits (B,T,V), got {tuple(logits.shape)}")
        if target.ndim != 2:
            raise ValueError(f"Expected target_ids (B,T), got {tuple(target.shape)}")
        B, T, _V = logits.shape
        if target.shape != (B, T):
            raise ValueError(f"target_ids shape mismatch: expected {(B, T)}, got {tuple(target.shape)}")

        # Generic binned accuracy path (for non-memory Table 2 rows, e.g. ICL-style tasks):
        # Provide `table2_bin` as (B,T) int64 with -1 for ignore.
        try:
            tb = batch["table2_bin"]
        except KeyError:
            tb = None
        if isinstance(tb, Tensor):
            if tb.shape != (B, T):
                raise ValueError(f"table2_bin shape mismatch: expected {(B, T)}, got {tuple(tb.shape)}")
            bins = tb.detach().cpu().long()
            pred_c = logits.argmax(dim=-1).detach().cpu().long()
            tgt_c = target.detach().cpu().long()

            n_bins = int(self.cfg.n_bins)
            if n_bins < 1:
                raise ValueError(f"n_bins must be >= 1, got {n_bins}")

            total = [0] * n_bins
            correct = [0] * n_bins

            for b in range(int(B)):
                for t in range(int(T)):
                    bi = int(bins[b, t].item())
                    if bi < 0:
                        continue
                    if bi >= n_bins:
                        raise ValueError(f"table2_bin out of range: {bi} for n_bins={n_bins}")
                    total[bi] += 1
                    if int(pred_c[b, t].item()) == int(tgt_c[b, t].item()):
                        correct[bi] += 1

            out_generic: dict[str, float] = {}
            worst_generic: float | None = None
            for i in range(n_bins):
                if total[i] > 0:
                    acc = float(correct[i]) / float(total[i])
                else:
                    acc = -1.0
                out_generic[f"acc/bin_{i}"] = float(acc)
                if total[i] > 0:
                    worst_generic = acc if worst_generic is None else float(min(worst_generic, acc))

            out_generic["acc/worst_bin"] = float(worst_generic) if worst_generic is not None else -1.0
            out_generic["collision/wrong_item_read_rate"] = -1.0
            return out_generic

        # Memory curriculum path: teacher bucket signals define read/query positions.
        try:
            input_ids = batch["input_ids"]
            rb = batch["mosaic_teacher_read_bucket"]
            wb = batch["mosaic_teacher_write_bucket"]
            wg = batch["mosaic_teacher_write_gate"]
        except KeyError as e:
            raise KeyError("Missing MOSAIC teacher signals required for Table 2 memory telemetry") from e
        if not isinstance(input_ids, Tensor):
            raise TypeError(f"batch['input_ids'] must be a Tensor, got {type(input_ids).__name__}")
        if not isinstance(rb, Tensor):
            raise TypeError(f"batch['mosaic_teacher_read_bucket'] must be a Tensor, got {type(rb).__name__}")
        if not isinstance(wb, Tensor):
            raise TypeError(f"batch['mosaic_teacher_write_bucket'] must be a Tensor, got {type(wb).__name__}")
        if not isinstance(wg, Tensor):
            raise TypeError(f"batch['mosaic_teacher_write_gate'] must be a Tensor, got {type(wg).__name__}")
        if input_ids.shape != (B, T):
            raise ValueError(f"input_ids shape mismatch: expected {(B, T)}, got {tuple(input_ids.shape)}")
        if wg.shape != (B, T):
            raise ValueError(f"mosaic_teacher_write_gate shape mismatch: expected {(B, T)}, got {tuple(wg.shape)}")

        rb0 = self._first_bucket(rb, B=B, T=T, name="mosaic_teacher_read_bucket")
        wb0 = self._first_bucket(wb, B=B, T=T, name="mosaic_teacher_write_bucket")

        # Predict next tokens.
        pred = logits.argmax(dim=-1)
        if pred.shape != (B, T):
            raise RuntimeError("argmax produced wrong shape for predictions")

        # Move small tensors to CPU for cheap Python-side bookkeeping.
        pred_c = pred.detach().cpu().long()
        tgt_c = target.detach().cpu().long()
        rb_c = rb0.detach().cpu().long()
        wb_c = wb0.detach().cpu().long()
        wg_c = wg.detach().cpu().float()

        n_bins = int(self.cfg.n_bins)
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        total = [0] * n_bins
        correct = [0] * n_bins
        reads_total = 0
        wrong_item = 0

        for b in range(int(B)):
            last_write_pos: dict[int, int] = {}
            for t in range(int(T)):
                if float(wg_c[b, t].item()) > 0.5:
                    buck = int(wb_c[b, t].item())
                    if buck >= 0:
                        last_write_pos[buck] = int(t)

                buck_r = int(rb_c[b, t].item())
                if buck_r >= 0:
                    wp = last_write_pos.get(buck_r)
                    if wp is None:
                        # A read request without any prior write is a data contract violation.
                        raise ValueError(f"Read requested for bucket={buck_r} but no prior write exists in sample")
                    dist = int(t) - int(wp)
                    if dist < 0:
                        raise ValueError("Computed negative distance for read vs write positions")
                    bin_idx = min(int(dist * n_bins // max(1, int(T))), n_bins - 1)
                    total[bin_idx] += 1

                    pred_tok = int(pred_c[b, t].item())
                    tgt_tok = int(tgt_c[b, t].item())
                    ok = pred_tok == tgt_tok
                    if ok:
                        correct[bin_idx] += 1
                    else:
                        wrong_item += 1
                    reads_total += 1

        out: dict[str, float] = {}
        worst: float | None = None
        for i in range(n_bins):
            if total[i] > 0:
                acc = float(correct[i]) / float(total[i])
            else:
                acc = -1.0
            out[f"acc/bin_{i}"] = float(acc)
            if total[i] > 0:
                worst = acc if worst is None else float(min(worst, acc))

        out["acc/worst_bin"] = float(worst) if worst is not None else -1.0

        if reads_total > 0:
            out["collision/wrong_item_read_rate"] = float(wrong_item) / float(reads_total)
        else:
            out["collision/wrong_item_read_rate"] = -1.0

        return out

    @staticmethod
    def _first_bucket(x: Tensor, *, B: int, T: int, name: str) -> Tensor:
        if x.ndim == 2:
            if x.shape != (B, T):
                raise ValueError(f"{name} shape mismatch: expected {(B, T)}, got {tuple(x.shape)}")
            return x.long()
        if x.ndim == 3:
            if x.shape[0] != B or x.shape[1] != T:
                raise ValueError(f"{name} shape mismatch: expected (B,T,H)={(B, T)}, got {tuple(x.shape)}")
            return x[:, :, 0].long()
        raise ValueError(f"{name} must have shape (B,T) or (B,T,H), got {tuple(x.shape)}")


@dataclass(frozen=True, slots=True)
class Table2SummaryWriter:
    """Write a writer-ready summary JSON for Table 2 runs."""

    filename_prefix: str = "table2_summary"

    def write(
        self,
        *,
        out_dir: Path,
        run_id: str,
        mem_buckets: int,
        mem_hashes: int,
        model_size: str,
        metrics: dict[str, float],
        n_bins: int,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        acc = []
        for i in range(int(n_bins)):
            k = f"acc/bin_{i}"
            if k not in metrics:
                raise KeyError(f"Missing {k!r} in metrics")
            acc.append(float(metrics[k]))
        if "acc/worst_bin" not in metrics:
            raise KeyError("Missing 'acc/worst_bin' in metrics")
        if "collision/wrong_item_read_rate" not in metrics:
            raise KeyError("Missing 'collision/wrong_item_read_rate' in metrics")

        payload = {
            "config": {
                "mem_buckets": int(mem_buckets),
                "mem_hashes": int(mem_hashes),
                "model_size": str(model_size),
            },
            "metrics": {
                "acc_per_bin": [float(x) for x in acc],
                "acc_worst_bin": float(metrics["acc/worst_bin"]),
                "collision_proxy": float(metrics["collision/wrong_item_read_rate"]),
            },
        }
        path = out_dir / f"{self.filename_prefix}_{str(run_id)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

