"""Perplexity microscope: batch-level traces + summary stats.

This is intended as a "sidecar" for the perplexity benchmark: it dumps
raw batch-level loss/token counts plus derived running statistics so you can
inspect spikes, variance, and convergence behavior post-hoc.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from benchmark.perplexity import PerplexityResult


def _safe_name(name: str) -> str:
    s = str(name).strip() or "model"
    # Keep filenames cross-platform friendly.
    for ch in ("/", "\\", ":", " ", "\t", "\n"):
        s = s.replace(ch, "_")
    return s


def _quantile_sorted(xs: list[float], q: float) -> float:
    """Nearest-rank quantile for a sorted list."""
    if not xs:
        return float("nan")
    if q <= 0.0:
        return float(xs[0])
    if q >= 1.0:
        return float(xs[-1])
    # Nearest-rank (1-indexed).
    k = int(math.ceil(q * len(xs))) - 1
    k = max(0, min(len(xs) - 1, k))
    return float(xs[k])


def compute_summary(result: PerplexityResult, *, topk: int = 25) -> dict[str, Any]:
    """Compute "microscope" summary stats from a perplexity run."""
    n = min(len(result.batch_loss_sums), len(result.batch_token_counts))
    avg_losses: list[float] = []
    rows: list[dict[str, Any]] = []

    cum_loss = 0.0
    cum_tokens = 0
    for i in range(int(n)):
        ls = float(result.batch_loss_sums[i])
        tc = int(result.batch_token_counts[i])
        if tc > 0:
            avg = ls / float(tc)
            avg_losses.append(float(avg))
        else:
            avg = float("nan")
        if tc > 0:
            cum_loss += float(ls)
            cum_tokens += int(tc)
            run_loss = cum_loss / float(cum_tokens) if cum_tokens > 0 else float("nan")
            run_ppl = math.exp(run_loss) if run_loss < 700.0 else float("inf")
        else:
            run_loss = float("nan")
            run_ppl = float("nan")
        rows.append(
            {
                "batch_idx": int(i),
                "token_count": int(tc),
                "loss_sum": float(ls),
                "avg_loss": float(avg),
                "running_avg_loss": float(run_loss),
                "running_ppl": float(run_ppl),
            }
        )

    avg_sorted = sorted(x for x in avg_losses if math.isfinite(float(x)))
    med = _quantile_sorted(avg_sorted, 0.50)
    p90 = _quantile_sorted(avg_sorted, 0.90)
    p95 = _quantile_sorted(avg_sorted, 0.95)
    p99 = _quantile_sorted(avg_sorted, 0.99)

    # Worst batches by avg loss (only finite).
    worst = sorted(
        (
            (int(r["batch_idx"]), float(r["avg_loss"]))
            for r in rows
            if math.isfinite(float(r["avg_loss"]))
        ),
        key=lambda t: t[1],
        reverse=True,
    )[: max(0, int(topk))]

    spike_ratio = float(p99 / med) if (math.isfinite(p99) and math.isfinite(med) and med > 0) else float("nan")

    return {
        "model_name": str(result.model_name),
        "perplexity": float(result.perplexity),
        "avg_loss": float(result.loss),
        "num_tokens": int(result.num_tokens),
        "num_batches": int(result.num_batches),
        "batch_rows": rows,
        "avg_loss_distribution": {
            "n_finite": int(len(avg_sorted)),
            "p50": float(med),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "p99_over_p50": float(spike_ratio),
        },
        "worst_batches_by_avg_loss": [
            {"batch_idx": int(i), "avg_loss": float(v)} for i, v in worst
        ],
    }


def write_microscope(
    *,
    output_dir: Path,
    result: PerplexityResult,
    prefix: str | None = None,
    topk: int = 25,
) -> dict[str, Path]:
    """Write batch-level raw data + summary JSON into output_dir."""
    out_dir = Path(output_dir) / "perplexity_microscope"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = _safe_name(str(prefix) if prefix else str(result.model_name))

    summary = compute_summary(result, topk=topk)
    rows = list(summary["batch_rows"])

    # Batch CSV: easy to load in pandas/Excel.
    csv_path = out_dir / f"{name}_batches.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_idx", "token_count", "loss_sum", "avg_loss", "running_avg_loss", "running_ppl"])
        for r in rows:
            w.writerow(
                [
                    int(r["batch_idx"]),
                    int(r["token_count"]),
                    float(r["loss_sum"]),
                    float(r["avg_loss"]),
                    float(r["running_avg_loss"]),
                    float(r["running_ppl"]),
                ]
            )

    # Summary JSON: quantiles + worst batches.
    json_path = out_dir / f"{name}_summary.json"
    # Avoid duplicating the entire batch table in JSON if it’s huge; keep only the summary + worst batches.
    json_summary = dict(summary)
    json_summary.pop("batch_rows", None)
    json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    return {
        f"perplexity_microscope/{name}_batches.csv": csv_path,
        f"perplexity_microscope/{name}_summary.json": json_path,
    }

