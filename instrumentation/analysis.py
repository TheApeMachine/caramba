"""Post-run analysis figure generation.

Why this exists:
- Training emits JSONL for machines and rich logs for humans.
- A quick PNG summary (loss curves, key metrics) is a convenient artifact for
  browsing runs without writing custom notebooks.

This module is dependency-gated:
- If matplotlib isn't installed, generation becomes a no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from console import logger


def _try_import_pyplot() -> Any | None:
    try:
        if importlib.util.find_spec("matplotlib") is None:
            return None
        # Force a non-interactive backend so tests/CI don't require a GUI.
        mpl = importlib.import_module("matplotlib")
        try:
            mpl.use("Agg")  # type: ignore[attr-defined]
        except Exception:
            logger.error("Failed to use Agg backend, continuing")
        return importlib.import_module("matplotlib.pyplot")
    except (ImportError, ModuleNotFoundError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON line, continuing: {line}")
                continue
    except OSError:
        logger.error(f"Failed to read JSONL file, continuing: {path}")
        return []
    return events


def generate_analysis_png(train_jsonl_path: str | Path, out_png: str | Path) -> None:
    """Generate a simple PNG summary from `train.jsonl`.

    The figure is intentionally minimal:
    - Blockwise/global loss curves (if present)
    - Verification metrics (if present)
    """

    plt = _try_import_pyplot()
    if plt is None:
        return

    src = Path(train_jsonl_path)
    dst = Path(out_png)
    events = _read_jsonl(src)
    if not events:
        return

    # Collect timeseries.
    bw_steps: list[int] = []
    bw_loss: list[float] = []
    g_steps: list[int] = []
    g_loss: list[float] = []
    v_steps: list[int] = []
    v_loss: list[float] = []

    verify: dict[str, float] = {}

    for ev in events:
        if str(ev.get("type", "")) != "metrics":
            continue
        phase = str(ev.get("phase", ""))
        step = ev.get("step", None)
        data = ev.get("data", {})
        if not isinstance(data, dict):
            logger.error(f"Failed to parse metrics, continuing: {data}")
            continue
        metrics = data.get("metrics", {})
        if not isinstance(metrics, dict):
            logger.error(f"Failed to parse metrics, continuing: {metrics}")
            continue
        loss = metrics.get("loss", None)

        if phase == "blockwise" and isinstance(step, int) and isinstance(loss, (int, float)):
            bw_steps.append(int(step))
            bw_loss.append(float(loss))
        elif phase == "global" and isinstance(step, int) and isinstance(loss, (int, float)):
            g_steps.append(int(step))
            g_loss.append(float(loss))
        elif phase == "eval_global" and isinstance(step, int):
            vl = metrics.get("val_loss", None)
            if isinstance(vl, (int, float)):
                v_steps.append(int(step))
                v_loss.append(float(vl))
        elif phase.startswith("verify") and isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    verify[f"{phase}/{k}"] = float(v)
                elif isinstance(v, str):
                    # Parse stringified floats like "0.012345"
                    try:
                        verify[f"{phase}/{k}"] = float(v)
                    except ValueError:
                        logger.error(f"Failed to parse verify metric, continuing: {v}")

    # Plot.
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title("caramba run summary")
    if bw_steps:
        ax.plot(bw_steps, bw_loss, label="blockwise loss")
    if g_steps:
        ax.plot(g_steps, g_loss, label="global loss")
    if v_steps:
        ax.plot(v_steps, v_loss, label="val loss")
    if bw_steps or g_steps:
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()

    # Add verify metrics as text box.
    if verify:
        lines = [f"{k}: {v:.6g}" for k, v in sorted(verify.items())]
        txt = "\n".join(lines[:12])
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.error(f"Failed to create parent directory, continuing: {dst.parent}")
    try:
        fig.tight_layout()
    except Exception:
        logger.error(f"Failed to save figure, continuing: {dst}")
    try:
        fig.savefig(str(dst), dpi=150)
    except Exception:
        logger.error(f"Failed to save figure, continuing: {dst}")
    try:
        plt.close(fig)
    except Exception:
        logger.error(f"Failed to close figure, continuing: {dst}")

