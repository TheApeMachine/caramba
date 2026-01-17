"""
Visualization module for weighted scoring results.

Generates matplotlib visualizations for:
- Hard vs Soft vs Weighted accuracy comparison
- Difficulty breakdown by model
- Match type distribution
- Visual ranking table
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
    figsize: tuple[float, float] = (10, 6),
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

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, hard_accs, width, label='Hard (EXACT)', color='#2ecc71')
    bars2 = ax.bar(x, soft_accs, width, label='Soft (EXACT+CONTAINED)', color='#3498db')
    bars3 = ax.bar(x + width, weighted_accs, width, label='Weighted', color='#9b59b6')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)

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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_difficulty_breakdown(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Score Contribution by Difficulty",
    figsize: tuple[float, float] = (10, 6),
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

    fig, ax = plt.subplots(figsize=figsize)

    # Stacked bars
    bars1 = ax.bar(x, easy_scores, width, label='Easy (1x)', color='#27ae60')
    bars2 = ax.bar(x, medium_scores, width, bottom=easy_scores, label='Medium (2x)', color='#f39c12')
    bars3 = ax.bar(x, hard_scores, width,
                   bottom=[e + m for e, m in zip(easy_scores, medium_scores)],
                   label='Hard (3x)', color='#e74c3c')

    ax.set_xlabel('Model')
    ax.set_ylabel('Weighted Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right')

    # Add total labels on top
    totals = [e + m + h for e, m, h in zip(easy_scores, medium_scores, hard_scores)]
    for i, total in enumerate(totals):
        ax.annotate(f'{total:.1f}',
                   xy=(x[i], total),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_match_type_distribution(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Match Type Distribution",
    figsize: tuple[float, float] = (10, 6),
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

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x, exact_counts, width, label='EXACT', color='#27ae60')
    bars2 = ax.bar(x, contained_counts, width, bottom=exact_counts, label='CONTAINED', color='#f39c12')
    bars3 = ax.bar(x, none_counts, width,
                   bottom=[e + c for e, c in zip(exact_counts, contained_counts)],
                   label='NONE', color='#e74c3c')

    ax.set_xlabel('Model')
    ax.set_ylabel('Test Count')
    ax.set_title(title)
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_weighted_ranking_table(
    summaries: dict[str, WeightedModelSummary],
    output_path: Path,
    title: str = "Weighted Scoring Rankings",
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """
    Create visual table showing rankings per metric.

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

    # Prepare data
    data = []
    for m in model_names:
        s = summaries[m]
        data.append([
            m,
            f"{s.hard_accuracy * 100:.1f}%",
            f"{s.soft_accuracy * 100:.1f}%",
            f"{s.weighted_accuracy * 100:.1f}%",
            f"{s.weighted_score_sum:.1f}/{s.weighted_score_max:.1f}",
        ])

    # Sort by weighted accuracy descending
    data_sorted = sorted(data, key=lambda x: float(x[3].replace('%', '')), reverse=True)

    # Add rank
    for i, row in enumerate(data_sorted):
        row.insert(0, str(i + 1))

    columns = ['Rank', 'Model', 'Hard %', 'Soft %', 'Weighted %', 'Score']

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=data_sorted,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(columns),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Color cells based on values (green=good, red=bad)
    for row_idx, row_data in enumerate(data_sorted):
        # Weighted accuracy column (index 4)
        weighted_val = float(row_data[4].replace('%', ''))
        if weighted_val >= 70:
            table[(row_idx + 1, 4)].set_facecolor('#a8e6cf')
        elif weighted_val >= 50:
            table[(row_idx + 1, 4)].set_facecolor('#ffeead')
        else:
            table[(row_idx + 1, 4)].set_facecolor('#ffaaa5')

        # Rank column highlighting
        if row_idx == 0:
            table[(row_idx + 1, 0)].set_facecolor('#ffd700')  # Gold
        elif row_idx == 1:
            table[(row_idx + 1, 0)].set_facecolor('#c0c0c0')  # Silver
        elif row_idx == 2:
            table[(row_idx + 1, 0)].set_facecolor('#cd7f32')  # Bronze

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_weighted_visualizations(
    summaries: dict[str, WeightedModelSummary],
    output_dir: Path,
    prefix: str = "",
) -> dict[str, Path]:
    """
    Generate all weighted scoring visualizations.

    Args:
        summaries: Dict of model_name -> WeightedModelSummary
        output_dir: Directory to save figures
        prefix: Optional prefix for filenames

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

    # 4. Ranking table
    path = output_dir / f"{prefix}weighted_ranking_table.png"
    plot_weighted_ranking_table(summaries, path)
    paths["weighted_ranking_table"] = path

    return paths
