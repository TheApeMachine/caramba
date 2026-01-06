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
    lines.append("% Auto-generated from W&B export CSV.")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{A100 training summary (latest logged step; W\\&B export).}")
    lines.append("\\label{tab:a100_training_summary_10k}")
    lines.append("\\begin{tabular}{@{}lrrr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Run} & \\textbf{Last step} & \\textbf{Loss} & \\textbf{PPL proxy} \\\\")
    lines.append("\\midrule")
    for name, last_step, last_loss in rows:
        lines.append(f"{name} & {last_step} & {last_loss:.4f} & {_ppl(last_loss):.1f} \\\\")
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
    (here / "A100-1b-10k-loss.png").write_bytes(b"")  # ensure file exists even if save fails
    fig.savefig(here / f"{args.prefix}-loss.png", dpi=200, bbox_inches="tight")
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

    print(f"[ok] wrote: {here/f'{args.prefix}-loss.png'}")
    print(f"[ok] wrote: {here/f'{args.prefix}-ppl.png'}")
    print(f"[ok] wrote: {here/'A100-1b-10k-training_summary.csv'}")
    print(f"[ok] wrote: {here/'A100-1b-10k-training_summary.tex'}")


if __name__ == "__main__":
    main()

