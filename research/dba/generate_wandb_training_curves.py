from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass(frozen=True)
class Series:
    name: str
    xs: list[int]
    ys: list[float]


@dataclass(frozen=True)
class SeriesBand:
    """Mean series with optional min/max band."""

    name: str
    xs: list[int]
    ys: list[float]
    ys_min: list[float] | None = None
    ys_max: list[float] | None = None


def _to_int(x: str) -> int:
    return int(float(str(x).strip().strip('"')))


def _to_float_or_none(x: str) -> float | None:
    s = str(x).strip().strip('"')
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_wandb_export(path: Path) -> tuple[list[int], dict[str, list[float | None]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV has no header")
        if "Step" not in r.fieldnames:
            raise ValueError("CSV missing required column: Step")

        steps: list[int] = []
        cols: dict[str, list[float | None]] = {k: [] for k in r.fieldnames if k != "Step"}
        for row in r:
            steps.append(_to_int(row["Step"]))
            for k in cols.keys():
                cols[k].append(_to_float_or_none(row.get(k, "")))
    return steps, cols


def _extract_series(
    steps: list[int], cols: dict[str, list[float | None]], *, suffix: str = ":train - loss"
) -> list[Series]:
    out: list[Series] = []
    for k, vs in cols.items():
        if not str(k).endswith(suffix):
            continue
        xs: list[int] = []
        ys: list[float] = []
        for s, v in zip(steps, vs, strict=True):
            if v is None:
                continue
            xs.append(int(s))
            ys.append(float(v))
        out.append(Series(name=str(k).removesuffix(suffix), xs=xs, ys=ys))
    if not out:
        raise ValueError(f"No series found ending with {suffix!r}")
    return out


def _extract_series_band(
    steps: list[int],
    cols: dict[str, list[float | None]],
    *,
    suffix: str,
) -> list[SeriesBand]:
    """Extract mean series plus optional __MIN/__MAX columns."""
    out: list[SeriesBand] = []
    keys = list(cols.keys())
    for k in keys:
        if not str(k).endswith(suffix):
            continue
        base = str(k)
        kmin = base + "__MIN"
        kmax = base + "__MAX"

        xs: list[int] = []
        ys: list[float] = []
        ys_min: list[float] = []
        ys_max: list[float] = []
        have_min = kmin in cols
        have_max = kmax in cols

        vs = cols[base]
        vmin = cols.get(kmin, [])
        vmax = cols.get(kmax, [])
        for i, (s, v) in enumerate(zip(steps, vs, strict=True)):
            if v is None:
                continue
            xs.append(int(s))
            ys.append(float(v))
            if have_min and i < len(vmin) and vmin[i] is not None:
                ys_min.append(float(vmin[i]))  # type: ignore[arg-type]
            else:
                ys_min.append(float(v))
            if have_max and i < len(vmax) and vmax[i] is not None:
                ys_max.append(float(vmax[i]))  # type: ignore[arg-type]
            else:
                ys_max.append(float(v))

        out.append(
            SeriesBand(
                name=str(base).removesuffix(suffix),
                xs=xs,
                ys=ys,
                ys_min=ys_min if (have_min or have_max) else None,
                ys_max=ys_max if (have_min or have_max) else None,
            )
        )
    if not out:
        raise ValueError(f"No series found ending with {suffix!r}")
    return out


def _ppl(loss: float) -> float:
    # Guard overflow; this is a *proxy* based on exp(nll).
    if loss > 50:
        return float("inf")
    return float(math.exp(float(loss)))


def _write_summary_table(
    out_dir: Path,
    *,
    loss_series: list[Series],
) -> None:
    def _latex_escape_text(s: str) -> str:
        # Minimal escaping for LaTeX table text cells.
        t = str(s)
        t = t.replace("\\", r"\textbackslash{}")
        t = t.replace("&", r"\&")
        t = t.replace("%", r"\%")
        t = t.replace("#", r"\#")
        t = t.replace("_", r"\_")
        t = t.replace("{", r"\{").replace("}", r"\}")
        return t

    rows: list[tuple[str, int, float]] = []
    for s in loss_series:
        if not s.xs or not s.ys:
            continue
        rows.append((s.name, int(s.xs[-1]), float(s.ys[-1])))

    rows.sort(key=lambda r: r[0])

    # CSV
    csv_path = out_dir / "A100-1b-10k-training_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "last_step", "last_loss", "last_ppl_proxy"])
        for name, last_step, last_loss in rows:
            w.writerow([name, last_step, f"{last_loss:.6f}", f"{_ppl(last_loss):.3f}"])

    # LaTeX (for easy \input)
    tex_path = out_dir / "A100-1b-10k-training_summary.tex"
    lines: list[str] = []
    # Help editors/language-servers lint this fragment in the context of paper.tex.
    lines.append("% !TEX root = paper.tex")
    lines.append("% Auto-generated from W&B export CSV.")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{A100 training summary (latest logged step; W\\&B export).}")
    # NOTE: avoid '_' in labels; it triggers 'Missing $ inserted' in some toolchains.
    lines.append("\\label{tab:a100-training-summary-10k}")
    lines.append("\\begin{tabular}{@{}lrrr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Run} & \\textbf{Last step} & \\textbf{Loss} & \\textbf{PPL proxy} \\\\")
    lines.append("\\midrule")
    for name, last_step, last_loss in rows:
        lines.append(f"{_latex_escape_text(name)} & {last_step} & {last_loss:.4f} & {_ppl(last_loss):.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to W&B export CSV. If omitted, uses newest research/dba/wandb_export*.csv.",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="A100-1b-10k",
        help="Output filename prefix (without extension).",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Warmup length (steps) for spike marker (default: 2000).",
    )
    args = ap.parse_args()

    if str(args.csv).strip():
        csv_path = Path(str(args.csv)).expanduser()
        if not csv_path.is_absolute():
            csv_path = (here / csv_path).resolve()
    else:
        candidates = list(here.glob("wandb_export*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No wandb_export*.csv found in {here}")
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        csv_path = candidates[0]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    steps, cols = _read_wandb_export(csv_path)
    loss_series = _extract_series(steps, cols, suffix=":train - loss")

    # Matplotlib is in project deps; import lazily so the rest remains light.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in loss_series:
        ax.plot(s.xs, s.ys, linewidth=2, label=s.name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (NLL)")
    ax.set_title("A100 training loss (W&B export)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = here / f"{args.prefix}-loss.png"
    out_path.write_bytes(b"")  # ensure file exists even if save fails
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- PPL proxy plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in loss_series:
        ax.plot(s.xs, [_ppl(y) for y in s.ys], linewidth=2, label=s.name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity proxy (exp(loss))")
    ax.set_title("A100 training perplexity proxy (W&B export)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    (here / f"{args.prefix}-ppl.png").write_bytes(b"")
    fig.savefig(here / f"{args.prefix}-ppl.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Keep legacy names for paper inclusion, regardless of chosen prefix.
    _write_summary_table(here, loss_series=loss_series)

    # --- Optional: warmup-boundary spike diagnostics (loss band + LR) ---
    # If present, these CSVs should contain mean + __MIN/__MAX for loss/lr.
    loss_spike_files = sorted(here.glob("baseline_decoupled_10k_loss_spikes*.csv"))
    lr_spike_files = sorted(here.glob("baseline_decoupled_10k_lr_spikes*.csv"))
    if loss_spike_files and lr_spike_files:
        # Use configured warmup if provided; this avoids inferring from noisy LR logs.
        warmup_step = int(args.warmup_steps) if int(args.warmup_steps) > 0 else None

        # Merge bands across all provided spike CSVs (useful when you export multiple failed runs).
        loss_bands: list[SeriesBand] = []
        lr_bands: list[SeriesBand] = []
        for p in loss_spike_files:
            steps_l, cols_l = _read_wandb_export(p)
            loss_bands.extend(_extract_series_band(steps_l, cols_l, suffix=":train - loss"))
        for p in lr_spike_files:
            steps_r, cols_r = _read_wandb_export(p)
            lr_bands.extend(_extract_series_band(steps_r, cols_r, suffix=":train - lr"))

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        # Loss with min/max band
        for s in loss_bands:
            ax0.plot(s.xs, s.ys, linewidth=2, label=s.name)
            if s.ys_min is not None and s.ys_max is not None:
                # Make spikes obvious: show both fill and explicit max trace.
                ax0.fill_between(s.xs, s.ys_min, s.ys_max, alpha=0.18)
                ax0.plot(s.xs, s.ys_max, linewidth=1.0, alpha=0.35)
        ax0.set_ylabel("Loss (NLL)")
        ax0.set_title("Warmup-boundary stability (loss band + LR)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8, ncol=1)

        # LR with min/max band
        for s in lr_bands:
            ax1.plot(s.xs, s.ys, linewidth=2, label=s.name)
            if s.ys_min is not None and s.ys_max is not None:
                ax1.fill_between(s.xs, s.ys_min, s.ys_max, alpha=0.15)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Learning rate")
        ax1.grid(True, alpha=0.3)

        if warmup_step is not None:
            ax0.axvline(warmup_step, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
            ax1.axvline(warmup_step, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
            ax1.text(
                warmup_step,
                ax1.get_ylim()[1],
                " warmup end",
                va="top",
                ha="left",
                fontsize=9,
                alpha=0.8,
            )

        fig.tight_layout()
        fig.savefig(here / "A100-1b-10k-warmup_spikes.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[ok] wrote: {here/f'{args.prefix}-loss.png'}")
    print(f"[ok] wrote: {here/f'{args.prefix}-ppl.png'}")
    print(f"[ok] wrote: {here/'A100-1b-10k-training_summary.csv'}")
    print(f"[ok] wrote: {here/'A100-1b-10k-training_summary.tex'}")
    if (here / "A100-1b-10k-warmup_spikes.png").exists():
        print(f"[ok] wrote: {here/'A100-1b-10k-warmup_spikes.png'}")


if __name__ == "__main__":
    main()

