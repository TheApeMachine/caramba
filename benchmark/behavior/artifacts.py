from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.behavior.types import BehaviorResult, BehaviorSummary, CaseResult, MatchType


def write_behavior_artifacts(
    *,
    result: BehaviorResult,
    output_dir: Path,
    ppl_by_model: dict[str, float] | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # ---------------------------------------------------------------------
    # Raw log (JSONL) + normalized CSV: prompts, outputs, scores, logprobs...
    # ---------------------------------------------------------------------
    raw_jsonl = output_dir / "behavior_raw.jsonl"
    with open(raw_jsonl, "w", encoding="utf-8") as f:
        for cr in result.results:
            row = _case_to_raw_row(cr)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths[raw_jsonl.name] = raw_jsonl

    raw_csv = output_dir / "behavior_raw.csv"
    _write_raw_csv(raw_csv, result.results, model_names=list(result.summaries.keys()))
    paths[raw_csv.name] = raw_csv

    # ---------------------------------------------------------------------
    # Summary JSON (overall + per-category)
    # ---------------------------------------------------------------------
    summary_json = output_dir / "behavior_summary.json"
    summary_payload = _build_summary_payload(result=result, ppl_by_model=ppl_by_model)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    paths[summary_json.name] = summary_json

    # ---------------------------------------------------------------------
    # Summary CSV (overall) + per-category CSV
    # ---------------------------------------------------------------------
    overall_csv = output_dir / "behavior_overall.csv"
    _write_overall_csv(overall_csv, summary_payload)
    paths[overall_csv.name] = overall_csv

    by_cat_csv = output_dir / "behavior_by_category.csv"
    _write_by_category_csv(by_cat_csv, summary_payload)
    paths[by_cat_csv.name] = by_cat_csv

    # ---------------------------------------------------------------------
    # LaTeX tables
    # ---------------------------------------------------------------------
    overall_tex = output_dir / "behavior_table.tex"
    overall_tex.write_text(_render_overall_latex(summary_payload), encoding="utf-8")
    paths[overall_tex.name] = overall_tex

    by_cat_tex = output_dir / "behavior_by_category_table.tex"
    by_cat_tex.write_text(_render_by_category_latex(summary_payload), encoding="utf-8")
    paths[by_cat_tex.name] = by_cat_tex

    # ---------------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------------
    paths.update(_plot_overall_bars(output_dir, summary_payload))
    paths.update(_plot_category_heatmap(output_dir, summary_payload))
    if ppl_by_model:
        paths.update(_plot_pareto(output_dir, summary_payload))

    return paths


def _case_to_raw_row(cr: CaseResult) -> dict[str, Any]:
    case = cr.case
    out: dict[str, Any] = {
        "case_id": case.id,
        "category": case.category,
        "difficulty": case.difficulty.value,
        "kind": case.kind.value,
        "prompt": case.prompt,
        "expected": case.expected,
        "target_text": case.target_text,
        "choices": list(case.choices),
        "correct_index": case.correct_index,
        "scoring_policy": {
            "allow_contained": case.allow_contained,
            "contained_constraints": [c.value for c in case.contained_constraints],
            "disallow_contained_if_expected_in_prompt": case.disallow_contained_if_expected_in_prompt,
        },
        "metadata": case.metadata,
        "outputs": {},
    }
    for mn, mo in cr.outputs.items():
        out["outputs"][mn] = {
            "output_text": mo.output_text,
            "match_type": mo.match_type.value,
            "raw_score": mo.raw_score,
            "difficulty_weight": mo.difficulty_weight,
            "baseline_weight": mo.baseline_weight,
            "final_score": mo.final_score,
            "expected_logprob": mo.expected_logprob,
            "choice_logprob": (asdict(mo.choice_logprob) if mo.choice_logprob is not None else None),
        }
    return out


def _write_raw_csv(path: Path, results: list[CaseResult], model_names: list[str]) -> None:
    fields = [
        "case_id",
        "category",
        "difficulty",
        "kind",
        "expected",
        "target_text",
        "prompt",
    ]
    # Flatten per-model columns.
    for mn in model_names:
        fields.extend(
            [
                f"{mn}_output",
                f"{mn}_match",
                f"{mn}_final_score",
                f"{mn}_expected_logprob",
                f"{mn}_choice_picked",
                f"{mn}_choice_correct",
                f"{mn}_choice_margin_logprob",
            ]
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for cr in results:
            row: list[Any] = [
                cr.case.id,
                cr.case.category,
                cr.case.difficulty.value,
                cr.case.kind.value,
                cr.case.expected,
                cr.case.target_text or "",
                cr.case.prompt,
            ]
            for mn in model_names:
                mo = cr.outputs.get(mn)
                if mo is None:
                    raise RuntimeError(f"Missing model output for model={mn!r}, case={cr.case.id!r}.")
                clp = mo.choice_logprob
                row.extend(
                    [
                        mo.output_text,
                        mo.match_type.value,
                        mo.final_score,
                        mo.expected_logprob,
                        (clp.picked if clp is not None else ""),
                        (clp.correct if clp is not None else ""),
                        (clp.margin_logprob if clp is not None else None),
                    ]
                )
            w.writerow(row)


def _build_summary_payload(*, result: BehaviorResult, ppl_by_model: dict[str, float] | None) -> dict[str, Any]:
    baseline = result.baseline_name
    if ppl_by_model is not None:
        if baseline not in ppl_by_model:
            raise RuntimeError(
                f"Behavior artifacts require ppl for baseline model {baseline!r}, but it is missing."
            )
        if any((m not in ppl_by_model) for m in result.summaries.keys()):
            missing = [m for m in result.summaries.keys() if m not in ppl_by_model]
            raise RuntimeError(f"Behavior artifacts: ppl missing for models: {missing!r}")

    base_ppl = float(ppl_by_model[baseline]) if ppl_by_model is not None else None
    by_model: dict[str, Any] = {}

    for mn, s in result.summaries.items():
        d = asdict(s)
        if ppl_by_model is not None and base_ppl is not None:
            ppl = float(ppl_by_model[mn])
            delta = (ppl / base_ppl) - 1.0
            # Penalize only worse-than-baseline perplexity (do not reward negative deltas).
            adj = float(s.weighted_accuracy) / (1.0 + max(0.0, float(delta)))
            d["ppl"] = ppl
            d["ppl_delta_vs_baseline"] = float(delta)
            d["ppl_adjusted_score"] = float(adj)
        by_model[mn] = d

    # Per-category breakdown (weighted_accuracy only + counts)
    by_category: dict[str, Any] = {}
    cats = sorted({c.category for c in result.cases})
    for cat in cats:
        by_category[cat] = {}
        cat_cases = [cr for cr in result.results if cr.case.category == cat]
        if not cat_cases:
            raise RuntimeError(f"Category {cat!r} has no cases in results.")
        for mn in result.summaries.keys():
            scores = [float(cr.outputs[mn].final_score) for cr in cat_cases]
            max_scores = [float(cr.outputs[mn].difficulty_weight) for cr in cat_cases]
            ws = float(sum(scores))
            wmax = float(sum(max_scores))
            by_category[cat][mn] = {
                "n": len(cat_cases),
                "score_sum": ws,
                "score_max": wmax,
                "weighted_accuracy": (ws / wmax if wmax > 0 else 0.0),
            }

    return {
        "suite_id": result.suite_id,
        "baseline_name": result.baseline_name,
        "models": by_model,
        "by_category": by_category,
        "n_cases": len(result.cases),
        "suite_config": result.suite_config,
    }


def _write_overall_csv(path: Path, payload: dict[str, Any]) -> None:
    models = payload["models"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "n",
                "exact",
                "contained",
                "none",
                "hard_accuracy",
                "soft_accuracy",
                "weighted_accuracy",
                "score_sum",
                "score_max",
                "ppl",
                "ppl_delta_vs_baseline",
                "ppl_adjusted_score",
            ]
        )
        for mn, d in models.items():
            w.writerow(
                [
                    mn,
                    d.get("n"),
                    d.get("exact"),
                    d.get("contained"),
                    d.get("none"),
                    d.get("hard_accuracy"),
                    d.get("soft_accuracy"),
                    d.get("weighted_accuracy"),
                    d.get("score_sum"),
                    d.get("score_max"),
                    d.get("ppl"),
                    d.get("ppl_delta_vs_baseline"),
                    d.get("ppl_adjusted_score"),
                ]
            )


def _write_by_category_csv(path: Path, payload: dict[str, Any]) -> None:
    models = list(payload["models"].keys())
    cats = list(payload["by_category"].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "n"] + [f"{m}_weighted_accuracy" for m in models])
        for cat in cats:
            n = int(next(iter(payload["by_category"][cat].values()))["n"])
            row = [cat, n]
            for m in models:
                row.append(payload["by_category"][cat][m]["weighted_accuracy"])
            w.writerow(row)


def _latex_escape(s: str) -> str:
    t = str(s)
    t = t.replace("\\", r"\textbackslash{}")
    t = t.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    t = t.replace("{", r"\{").replace("}", r"\}")
    return t


def _render_overall_latex(payload: dict[str, Any]) -> str:
    models = payload["models"]
    # Best-of bolding
    best_weighted = max((float(v.get("weighted_accuracy", 0.0)), k) for k, v in models.items())[1]
    best_adj = None
    if any(models[m].get("ppl_adjusted_score") is not None for m in models):
        best_adj = max(
            (float(v.get("ppl_adjusted_score") or 0.0), k) for k, v in models.items()
        )[1]

    lines: list[str] = []
    lines.append("% Auto-generated behavior benchmark table")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Behavior benchmark (completion-style) results.}")
    lines.append(r"\label{tab:behavior}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Hard & Soft & Weighted & $\Delta$PPL & PPL-adj \\")
    lines.append(r"\midrule")
    base = payload["baseline_name"]
    base_ppl = models[base].get("ppl") if base in models else None
    for mn, d in models.items():
        hard = 100.0 * float(d.get("hard_accuracy") or 0.0)
        soft = 100.0 * float(d.get("soft_accuracy") or 0.0)
        wacc = 100.0 * float(d.get("weighted_accuracy") or 0.0)
        dppl = float(d.get("ppl_delta_vs_baseline") or 0.0) * 100.0 if d.get("ppl_delta_vs_baseline") is not None else float("nan")
        padj = 100.0 * float(d.get("ppl_adjusted_score") or 0.0) if d.get("ppl_adjusted_score") is not None else float("nan")

        w_str = f"{wacc:.1f}\\%"
        if mn == best_weighted:
            w_str = rf"\textbf{{{w_str}}}"
        a_str = f"{padj:.1f}\\%" if not math.isnan(padj) else "--"
        if best_adj is not None and mn == best_adj:
            a_str = rf"\textbf{{{a_str}}}"
        dppl_str = f"{dppl:+.2f}\\%" if not math.isnan(dppl) else "--"
        lines.append(
            rf"{_latex_escape(mn)} & {hard:.1f}\% & {soft:.1f}\% & {w_str} & {dppl_str} & {a_str} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _render_by_category_latex(payload: dict[str, Any]) -> str:
    models = list(payload["models"].keys())
    cats = list(payload["by_category"].keys())
    lines: list[str] = []
    lines.append("% Auto-generated behavior benchmark by-category table")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Behavior benchmark results by category (weighted accuracy).}")
    lines.append(r"\label{tab:behavior-by-category}")
    cols = "l" + ("r" * len(models))
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    lines.append("Category & " + " & ".join(_latex_escape(m) for m in models) + r" \\")
    lines.append(r"\midrule")
    for cat in cats:
        row = [rf"{_latex_escape(cat)}"]
        for m in models:
            v = float(payload["by_category"][cat][m]["weighted_accuracy"]) * 100.0
            row.append(f"{v:.1f}\\%")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _plot_overall_bars(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    models = payload["models"]
    names = list(models.keys())
    vals = [100.0 * float(models[n].get("weighted_accuracy") or 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(names)), 4.2))
    bars = ax.bar(names, vals, color="tab:blue", alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("Behavior weighted accuracy (%) ↑")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = output_dir / "behavior_weighted_overall.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {p.name: p}


def _plot_category_heatmap(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    models = list(payload["models"].keys())
    cats = list(payload["by_category"].keys())
    M = np.zeros((len(cats), len(models)), dtype=np.float32)
    for i, cat in enumerate(cats):
        for j, m in enumerate(models):
            M[i, j] = float(payload["by_category"][cat][m]["weighted_accuracy"]) * 100.0

    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(models) + 4.0), max(5.0, 0.35 * len(cats) + 2.0)))
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_yticks(list(range(len(cats))))
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_xlabel("model")
    ax.set_ylabel("category")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="weighted accuracy (%)")
    fig.tight_layout()
    p = output_dir / "behavior_by_category_heatmap.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {p.name: p}


def _plot_pareto(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    models = payload["models"]
    base = payload["baseline_name"]
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for mn, d in models.items():
        if d.get("ppl_delta_vs_baseline") is None:
            continue
        xs.append(100.0 * float(d.get("ppl_delta_vs_baseline")))
        ys.append(100.0 * float(d.get("weighted_accuracy") or 0.0))
        labels.append(str(mn))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(xs, ys, s=55, alpha=0.85)
    for x, y, lab in zip(xs, ys, labels):
        ax.text(float(x) + 0.05, float(y) + 0.2, str(lab), fontsize=9, alpha=0.9)
    ax.axvline(0.0, color="black", alpha=0.15, linewidth=1.0)
    ax.set_xlabel("PPL delta vs baseline (%) (lower is better) →")
    ax.set_ylabel("Behavior weighted accuracy (%) ↑")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = output_dir / "behavior_pareto_ppl_vs_behavior.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {p.name: p}

