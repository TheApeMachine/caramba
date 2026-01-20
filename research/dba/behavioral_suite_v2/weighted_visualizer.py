"""
Visualization module for weighted scoring results.

Generates matplotlib visualizations for:
- Hard vs Soft vs Weighted accuracy comparison
- Difficulty breakdown by model
- Match type distribution
- Category breakdown comparison
- Visual ranking table

Also generates LaTeX tables for paper inclusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .weighted_scoring import (
    WeightedScorer,
    WeightedModelSummary,
    MatchType,
)


def plot_weighted_scores_comparison(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Weighted Scores Comparison",
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """
    Create grouped bar chart comparing hard, soft, and weighted accuracy per model.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size in inches
    """
    if not HAS_MATPLOTLIB:
        return

    model_names = list(summaries.keys())
    n_models = len(model_names)

    if n_models == 0:
        return

    hard_accs = [summaries[m].hard_accuracy * 100 for m in model_names]
    soft_accs = [summaries[m].soft_accuracy * 100 for m in model_names]
    weighted_accs = [summaries[m].weighted_accuracy * 100 for m in model_names]

    x = np.arange(n_models)
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars1 = ax.bar(x - width, hard_accs, width, label='Hard (EXACT)', color='#2ecc71')
    bars2 = ax.bar(x, soft_accs, width, label='Soft (EXACT+CONTAINED)', color='#3498db')
    bars3 = ax.bar(x + width, weighted_accs, width, label='Weighted', color='#9b59b6')

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    # No title (paper-friendly).
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)  # Extra room for labels

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_difficulty_breakdown(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Score Contribution by Baseline-Defined Difficulty",
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """
    Create stacked bar chart showing weighted score contribution by difficulty tier.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size in inches
    """
    if not HAS_MATPLOTLIB:
        return

    model_names = list(summaries.keys())
    n_models = len(model_names)

    if n_models == 0:
        return

    # Extract scores by difficulty for each model
    easy_scores = []
    medium_scores = []
    hard_scores = []

    for m in model_names:
        by_diff = summaries[m].by_difficulty
        easy_data = by_diff.get("easy")
        medium_data = by_diff.get("medium")
        hard_data = by_diff.get("hard")

        easy_scores.append(easy_data.weighted_score if easy_data else 0)
        medium_scores.append(medium_data.weighted_score if medium_data else 0)
        hard_scores.append(hard_data.weighted_score if hard_data else 0)

    x = np.arange(n_models)
    width = 0.6

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Stacked bars
    # NOTE: Difficulty here is baseline-defined:
    # - Easy (1x): baseline EXACT
    # - Medium (2x): baseline CONTAINED
    # - Hard (3x): baseline NONE
    bars1 = ax.bar(x, easy_scores, width, label='Baseline easy (1x)', color='#27ae60')
    bars2 = ax.bar(x, medium_scores, width, bottom=easy_scores, label='Baseline medium (2x)', color='#f39c12')
    bars3 = ax.bar(x, hard_scores, width,
                   bottom=[e + m for e, m in zip(easy_scores, medium_scores)],
                   label='Baseline hard (3x)', color='#e74c3c')

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Weighted Score', fontsize=11)
    # No title (paper-friendly).
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right')

    # Add total labels on top
    totals = [e + m + h for e, m, h in zip(easy_scores, medium_scores, hard_scores)]
    max_total = max(totals) if totals else 0
    # Ensure a minimum ylim to avoid UserWarning when all scores are 0
    ax.set_ylim(0, max(max_total * 1.15, 1.0))

    for i, total in enumerate(totals):
        ax.annotate(f'{total:.1f}',
                   xy=(x[i], total),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_match_type_distribution(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Match Type Distribution",
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """
    Create stacked bar chart showing EXACT / CONTAINED / NONE counts per model.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size in inches
    """
    if not HAS_MATPLOTLIB:
        return

    model_names = list(summaries.keys())
    n_models = len(model_names)

    if n_models == 0:
        return

    exact_counts = [summaries[m].exact_count for m in model_names]
    contained_counts = [summaries[m].contained_count for m in model_names]
    none_counts = [summaries[m].none_count for m in model_names]

    x = np.arange(n_models)
    width = 0.6

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars1 = ax.bar(x, exact_counts, width, label='EXACT', color='#27ae60')
    bars2 = ax.bar(x, contained_counts, width, bottom=exact_counts, label='CONTAINED', color='#f39c12')
    bars3 = ax.bar(x, none_counts, width,
                   bottom=[e + c for e, c in zip(exact_counts, contained_counts)],
                   label='NONE', color='#e74c3c')

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Test Count', fontsize=11)
    # No title (paper-friendly).
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right')

    # Add total and percentage labels
    for i, m in enumerate(model_names):
        total = exact_counts[i] + contained_counts[i] + none_counts[i]
        if total > 0:
            pct = exact_counts[i] / total * 100
            ax.annotate(f'{pct:.0f}% exact',
                       xy=(x[i], total),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_category_breakdown(
    category_data: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
    title: str = "Accuracy by Category",
    figsize: tuple[float, float] = (14, 8),
) -> None:
    """
    Create grouped bar chart showing accuracy per category per model.

    Args:
        category_data: Dict of model_name -> category_name -> {"exact": float, "soft": float}
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size in inches
    """
    if not HAS_MATPLOTLIB:
        return

    model_names = list(category_data.keys())
    if not model_names:
        return

    # Get all categories
    all_categories: set[str] = set()
    for model_data in category_data.values():
        all_categories.update(model_data.keys())
    categories = sorted(all_categories)

    if not categories:
        return

    n_models = len(model_names)
    n_categories = len(categories)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    x = np.arange(n_categories)
    width = 0.8 / n_models

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    for i, model_name in enumerate(model_names):
        model_data = category_data[model_name]
        soft_accs = [model_data.get(cat, {}).get("soft", 0) * 100 for cat in categories]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, soft_accs, width, label=model_name, color=colors[i % len(colors)])

    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('Soft Accuracy (%)', fontsize=11)
    # No title (paper-friendly).
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_category_heatmap(
    category_data: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
    title: str = "Accuracy Heatmap by Category",
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """
    Create heatmap showing accuracy per category per model.

    Args:
        category_data: Dict of model_name -> category_name -> {"exact": float, "soft": float}
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size in inches
    """
    if not HAS_MATPLOTLIB:
        return

    model_names = list(category_data.keys())
    if not model_names:
        return

    # Get all categories
    all_categories: set[str] = set()
    for model_data in category_data.values():
        all_categories.update(model_data.keys())
    categories = sorted(all_categories)

    if not categories:
        return

    # Build matrix
    matrix = np.zeros((len(model_names), len(categories)))
    for i, model_name in enumerate(model_names):
        model_data = category_data[model_name]
        for j, cat in enumerate(categories):
            matrix[i, j] = model_data.get(cat, {}).get("soft", 0) * 100

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(model_names, fontsize=10)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    # No title (paper-friendly).

    # Add colorbar on side
    fig.colorbar(im, ax=ax, orientation='vertical', label='Soft Accuracy (%)', shrink=0.8)

    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_weighted_ranking_table(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Weighted Scoring Rankings",
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """
    Create a bar chart showing model rankings by weighted accuracy.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size in inches
    """
    if not HAS_MATPLOTLIB:
        return

    model_names = list(summaries.keys())
    if not model_names:
        return

    # Prepare and sort data
    data = []
    for m in model_names:
        s = summaries[m]
        data.append({
            "name": m,
            "weighted": s.weighted_accuracy * 100,
            "hard": s.hard_accuracy * 100,
            "soft": s.soft_accuracy * 100
        })

    # Sort by weighted accuracy descending
    data_sorted = sorted(data, key=lambda x: x["weighted"], reverse=True)
    names = [d["name"] for d in data_sorted]
    weighted = [d["weighted"] for d in data_sorted]
    hard = [d["hard"] for d in data_sorted]
    soft = [d["soft"] for d in data_sorted]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    x = np.arange(len(names))
    width = 0.6

    # Plot weighted as main bars
    bars = ax.bar(x, weighted, width, color='#9b59b6', alpha=0.8, label='Weighted Accuracy')
    
    # Add markers for hard and soft
    ax.scatter(x, hard, marker='s', color='#2ecc71', s=50, label='Hard Accuracy', zorder=3)
    ax.scatter(x, soft, marker='o', color='#3498db', s=50, label='Soft Accuracy', zorder=3)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    # No title (paper-friendly).
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def generate_latex_ranking_table(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    caption: str = "Behavioral Test Results (Weighted Scoring)",
    label: str = "tab:behavioral_weighted",
) -> None:
    """
    Generate LaTeX table for paper inclusion.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_path: Path to save the .tex file
        caption: Table caption
        label: LaTeX label for referencing
    """
    model_names = list(summaries.keys())
    if not model_names:
        return

    # Sort by weighted accuracy descending
    sorted_models = sorted(
        model_names,
        key=lambda m: summaries[m].weighted_accuracy,
        reverse=True
    )

    lines = [
        "% Auto-generated LaTeX table",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & EXACT & CONTAINED & NONE & Hard \\% & Soft \\% & Weighted \\% \\\\",
        "\\midrule",
    ]

    for i, m in enumerate(sorted_models):
        s = summaries[m]
        bold_start = "\\textbf{" if i == 0 else ""
        bold_end = "}" if i == 0 else ""
        lines.append(
            f"{bold_start}{m}{bold_end} & "
            f"{s.exact_count} & {s.contained_count} & {s.none_count} & "
            f"{s.hard_accuracy * 100:.1f} & {s.soft_accuracy * 100:.1f} & "
            f"{bold_start}{s.weighted_accuracy * 100:.1f}{bold_end} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_latex_category_table(
    category_data: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
    caption: str = "Accuracy by Category",
    label: str = "tab:category_accuracy",
) -> None:
    """
    Generate LaTeX table showing accuracy per category per model.

    Args:
        category_data: Dict of model_name -> category_name -> {"exact": float, "soft": float}
        output_path: Path to save the .tex file
        caption: Table caption
        label: LaTeX label for referencing
    """
    model_names = list(category_data.keys())
    if not model_names:
        return

    # Get all categories
    all_categories: set[str] = set()
    for model_data in category_data.values():
        all_categories.update(model_data.keys())
    categories = sorted(all_categories)

    if not categories:
        return

    # Build header
    col_spec = "l" + "c" * len(categories)
    header_cats = " & ".join(categories)

    lines = [
        "% Auto-generated LaTeX table",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Model & {header_cats} \\\\",
        "\\midrule",
    ]

    for model_name in model_names:
        model_data = category_data[model_name]
        values = []
        for cat in categories:
            val = model_data.get(cat, {}).get("soft", 0) * 100
            values.append(f"{val:.0f}")
        lines.append(f"{model_name} & " + " & ".join(values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_all_weighted_visualizations(
    summaries: dict[str, WeightedModelSummary],
    output_dir: Path,
    prefix: str = "",
    category_data: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Path]:
    """
    Generate all weighted scoring visualizations and LaTeX tables.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_dir: Directory to save figures
        prefix: Optional prefix for filenames
        category_data: Optional category breakdown data for category visualizations

    Returns:
        Dict mapping visualization names to their file paths
    """
    if not HAS_MATPLOTLIB:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Weighted scores comparison (bar chart)
    path = output_dir / f"{prefix}weighted_scores_comparison.png"
    plot_weighted_scores_comparison(summaries, path)
    paths["weighted_scores_comparison"] = path

    # 2. Difficulty breakdown (stacked bar)
    path = output_dir / f"{prefix}difficulty_breakdown.png"
    plot_difficulty_breakdown(summaries, path)
    paths["difficulty_breakdown"] = path

    # 3. Match type distribution (stacked bar)
    path = output_dir / f"{prefix}match_type_distribution.png"
    plot_match_type_distribution(summaries, path)
    paths["match_type_distribution"] = path

    # 4. Ranking table (PNG)
    path = output_dir / f"{prefix}weighted_ranking_table.png"
    plot_weighted_ranking_table(summaries, path)
    paths["weighted_ranking_table"] = path

    # 5. Ranking table (LaTeX)
    path = output_dir / f"{prefix}weighted_ranking_table.tex"
    generate_latex_ranking_table(summaries, path)
    paths["weighted_ranking_table_latex"] = path

    # 6. Category breakdown (if data provided)
    if category_data:
        # Bar chart
        path = output_dir / f"{prefix}category_breakdown.png"
        plot_category_breakdown(category_data, path)
        paths["category_breakdown"] = path

        # Heatmap
        path = output_dir / f"{prefix}category_heatmap.png"
        plot_category_heatmap(category_data, path)
        paths["category_heatmap"] = path

        # LaTeX table
        path = output_dir / f"{prefix}category_table.tex"
        generate_latex_category_table(category_data, path)
        paths["category_table_latex"] = path

    return paths
