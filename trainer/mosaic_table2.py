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

from runtime.tensordict_utils import TensorDictBase


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
        tb = batch.get("table2_bin", None)  # type: ignore[attr-defined]
        out_generic: dict[str, float] | None = None
        has_valid_bins = False
        if isinstance(tb, Tensor):
            if tb.shape != (B, T):
                raise ValueError(f"table2_bin shape mismatch: expected {(B, T)}, got {tuple(tb.shape)}")
            bins = tb.detach().long()
            pred_c = logits.argmax(dim=-1).detach().long()
            tgt_c = target.detach().long()

            n_bins = int(self.cfg.n_bins)
            if n_bins < 1:
                raise ValueError(f"n_bins must be >= 1, got {n_bins}")

            valid_mask = bins >= 0
            has_valid_bins = bool(valid_mask.any())
            if has_valid_bins:
                max_bin = int(bins[valid_mask].max().item())
                if max_bin >= n_bins:
                    raise ValueError(f"table2_bin out of range: max={max_bin} for n_bins={n_bins}")

            bins_flat = bins[valid_mask].to(dtype=torch.long)
            pred_flat = pred_c[valid_mask]
            tgt_flat = tgt_c[valid_mask]
            correct_flat = (pred_flat == tgt_flat).to(dtype=torch.long)

            total_tensor = torch.zeros(n_bins, dtype=torch.long, device=bins.device)
            correct_tensor = torch.zeros(n_bins, dtype=torch.long, device=bins.device)
            total_tensor.scatter_add_(0, bins_flat, torch.ones_like(bins_flat))
            correct_tensor.scatter_add_(0, bins_flat, correct_flat)

            total = total_tensor.detach().cpu().tolist()
            correct = correct_tensor.detach().cpu().tolist()

            out_generic = {}
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
            if has_valid_bins:
                return out_generic

        # Memory curriculum path: teacher bucket signals define read/query positions.
        try:
            input_ids = batch["input_ids"]
            rb = batch["memblock_teacher_read_bucket"]
            wb = batch["memblock_teacher_write_bucket"]
            wg = batch["memblock_teacher_write_gate"]
        except KeyError as e:
            if out_generic is not None:
                return out_generic
            raise KeyError("Missing `table2_bin` and MOSAIC teacher signals required for Table 2 telemetry.") from e
        if not isinstance(input_ids, Tensor):
            raise TypeError(f"batch['input_ids'] must be a Tensor, got {type(input_ids).__name__}")
        if not isinstance(rb, Tensor):
            raise TypeError(f"batch['memblock_teacher_read_bucket'] must be a Tensor, got {type(rb).__name__}")
        if not isinstance(wb, Tensor):
            raise TypeError(f"batch['memblock_teacher_write_bucket'] must be a Tensor, got {type(wb).__name__}")
        if not isinstance(wg, Tensor):
            raise TypeError(f"batch['memblock_teacher_write_gate'] must be a Tensor, got {type(wg).__name__}")
        if input_ids.shape != (B, T):
            raise ValueError(f"input_ids shape mismatch: expected {(B, T)}, got {tuple(input_ids.shape)}")
        if wg.shape != (B, T):
            raise ValueError(f"memblock_teacher_write_gate shape mismatch: expected {(B, T)}, got {tuple(wg.shape)}")

        rb0 = self._first_bucket(rb, B=B, T=T, name="memblock_teacher_read_bucket")
        wb0 = self._first_bucket(wb, B=B, T=T, name="memblock_teacher_write_bucket")

        pred = logits.argmax(dim=-1)
        if pred.shape != (B, T):
            raise RuntimeError("argmax produced wrong shape for predictions")

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

        # NOTE: This path must be fast. In our curricula, teacher signals are sparse:
        # - writes occur at a handful of positions (e.g. n_demos)
        # - reads occur at a handful of positions (often 1)
        # Avoid scanning the full T dimension in Python.
        for b in range(int(B)):
            # Collect write positions and buckets.
            write_pos = (wg_c[b] > 0.5).nonzero(as_tuple=False).view(-1)
            last_write_pos: dict[int, int] = {}
            for t_idx in write_pos.tolist():
                buck = int(wb_c[b, int(t_idx)].item())
                if buck >= 0:
                    # If multiple writes hit same bucket, the last one wins.
                    last_write_pos[buck] = int(t_idx)

            # Collect read positions.
            read_pos = (rb_c[b] >= 0).nonzero(as_tuple=False).view(-1)
            for t_idx in read_pos.tolist():
                t_int = int(t_idx)
                buck_r = int(rb_c[b, t_int].item())
                if buck_r < 0:
                    continue
                wp = last_write_pos.get(buck_r)
                if wp is None:
                    # Some datasets may label reads for buckets that were not written
                    # in the same sample. Treat these as "no-op" for telemetry.
                    continue
                dist = int(t_int) - int(wp)
                if dist < 0:
                    # If the last write occurs after this read position, ignore.
                    continue
                bin_idx = min(int(dist * n_bins // max(1, int(T))), n_bins - 1)
                total[bin_idx] += 1

                pred_tok = int(pred_c[b, t_int].item())
                tgt_tok = int(tgt_c[b, t_int].item())
                if pred_tok == tgt_tok:
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
        out["collision/wrong_item_read_rate"] = (float(wrong_item) / float(reads_total)) if reads_total > 0 else -1.0
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

