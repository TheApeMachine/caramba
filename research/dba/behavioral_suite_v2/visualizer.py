"""
Visualization system for behavioral evaluation results.

Provides:
1. Attention heatmaps with shared colorbar (for fair comparison)
2. Per-category accuracy charts
3. Head-to-head comparison matrices
4. Failure mode breakdowns
5. Soft score distributions
6. Interactive HTML dashboard
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Import plotting libraries with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


from .scoring import SoftScore, TestScore


class AttentionVisualizer:
    """
    Visualizes attention patterns with proper normalization.

    CRITICAL: Uses shared colorbar with fixed vmin/vmax across models
    to enable fair visual comparison.
    """

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (12, 8),
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.figsize = figsize

    def plot_comparison(
        self,
        attention_a: np.ndarray,
        attention_b: np.ndarray,
        model_a_name: str,
        model_b_name: str,
        layer: int = 0,
        head: int = 0,
        tokens: list[str] | None = None,
        title: str = "",
        output_path: Path | None = None,
    ) -> None:
        """
        Plot side-by-side attention comparison with shared colorbar.

        Args:
            attention_a: Attention from model A [layers, heads, seq, seq]
            attention_b: Attention from model B [layers, heads, seq, seq]
            model_a_name: Display name for model A
            model_b_name: Display name for model B
            layer: Layer index to visualize
            head: Head index to visualize
            tokens: Token labels for axes
            title: Plot title
            output_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available for visualization")
            return

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Extract specific layer/head
        attn_a = attention_a[layer, head]
        attn_b = attention_b[layer, head]

        # Create shared normalization
        norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

        # Plot both with same normalization
        im_a = axes[0].imshow(attn_a, cmap=self.cmap, norm=norm, aspect='auto')
        axes[0].set_title(f"{model_a_name}\nLayer {layer}, Head {head}")

        im_b = axes[1].imshow(attn_b, cmap=self.cmap, norm=norm, aspect='auto')
        axes[1].set_title(f"{model_b_name}\nLayer {layer}, Head {head}")

        # Add token labels if provided
        if tokens:
            for ax in axes:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)

        # Add single shared colorbar
        cbar = fig.colorbar(im_b, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Attention Weight')

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_layer_summary(
        self,
        attention: np.ndarray,
        model_name: str,
        layers_to_plot: list[int] | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot attention patterns across multiple layers.

        Args:
            attention: Full attention tensor [layers, heads, seq, seq]
            model_name: Model name for title
            layers_to_plot: Which layers to show (None = sample uniformly)
            output_path: Where to save
        """
        if not HAS_MATPLOTLIB:
            return

        n_layers = attention.shape[0]
        if layers_to_plot is None:
            # Sample 6 layers uniformly
            layers_to_plot = list(range(0, n_layers, max(1, n_layers // 6)))[:6]

        n_plots = len(layers_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

        for i, layer_idx in enumerate(layers_to_plot):
            if i >= len(axes):
                break
            # Average over heads for this layer
            avg_attn = attention[layer_idx].mean(axis=0)
            im = axes[i].imshow(avg_attn, cmap=self.cmap, norm=norm, aspect='auto')
            axes[i].set_title(f"Layer {layer_idx}")

        # Hide unused axes
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        # Add shared colorbar
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

        fig.suptitle(f"{model_name} - Attention by Layer (averaged over heads)", fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()


class ResultsVisualizer:
    """
    Visualizes evaluation results and comparisons.
    """

    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        self.figsize = figsize
        if HAS_SEABORN:
            sns.set_theme(style="whitegrid")

    def plot_category_comparison(
        self,
        category_results: dict[str, dict[str, dict[str, Any]]],
        metric: str = "exact_match_rate",
        output_path: Path | None = None,
    ) -> None:
        """
        Plot per-category accuracy comparison across models.

        Args:
            category_results: From EvalResults.category_results
            metric: Which metric to plot
            output_path: Where to save
        """
        if not HAS_MATPLOTLIB:
            return

        categories = list(category_results.keys())
        models = set()
        for cat_data in category_results.values():
            models.update(cat_data.keys())
        models = sorted(models)

        # Build data matrix
        data = np.zeros((len(categories), len(models)))
        for i, cat in enumerate(categories):
            for j, model in enumerate(models):
                if model in category_results[cat]:
                    data[i, j] = category_results[cat][model].get(metric, 0)

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(categories))
        width = 0.8 / len(models)

        for j, model in enumerate(models):
            offset = (j - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, data[:, j], width, label=model)

        ax.set_xlabel('Category')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    def plot_head_to_head_matrix(
        self,
        comparisons: list[dict[str, Any]],
        output_path: Path | None = None,
    ) -> None:
        """
        Plot head-to-head win matrix.
        """
        if not HAS_MATPLOTLIB:
            return

        # Build model list and win matrix
        models = set()
        for comp in comparisons:
            models.add(comp['model_a'])
            models.add(comp['model_b'])
        models = sorted(models)

        n = len(models)
        win_matrix = np.zeros((n, n))

        for comp in comparisons:
            i = models.index(comp['model_a'])
            j = models.index(comp['model_b'])
            win_matrix[i, j] = comp['wins_a']
            win_matrix[j, i] = comp['wins_b']

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        if HAS_SEABORN:
            sns.heatmap(
                win_matrix,
                xticklabels=models,
                yticklabels=models,
                annot=True,
                fmt='.0f',
                cmap='Blues',
                ax=ax,
            )
        else:
            im = ax.imshow(win_matrix, cmap='Blues')
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(models)
            ax.set_yticklabels(models)
            # Add text annotations
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{win_matrix[i, j]:.0f}',
                           ha='center', va='center')

        ax.set_title('Head-to-Head Wins (row vs column)')
        ax.set_xlabel('Opponent')
        ax.set_ylabel('Model')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    def plot_soft_score_distribution(
        self,
        scores: dict[str, dict[str, TestScore]],
        output_path: Path | None = None,
    ) -> None:
        """
        Plot distribution of soft scores per model.
        """
        if not HAS_MATPLOTLIB:
            return

        models = list(scores.keys())
        score_names = [s.name for s in SoftScore]

        # Count scores per model
        counts = np.zeros((len(models), len(SoftScore)))
        for i, model in enumerate(models):
            for test_score in scores[model].values():
                score_idx = list(SoftScore).index(test_score.soft_score)
                counts[i, score_idx] += 1

        # Normalize to percentages
        totals = counts.sum(axis=1, keepdims=True)
        percentages = counts / totals * 100

        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        width = 0.6

        bottom = np.zeros(len(models))
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(SoftScore)))

        for j, score_name in enumerate(score_names):
            ax.bar(x, percentages[:, j], width, label=score_name,
                   bottom=bottom, color=colors[j])
            bottom += percentages[:, j]

        ax.set_xlabel('Model')
        ax.set_ylabel('Percentage of Tests')
        ax.set_title('Soft Score Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    def plot_failure_modes(
        self,
        summaries: dict[str, dict[str, Any]],
        output_path: Path | None = None,
    ) -> None:
        """
        Plot failure mode breakdown per model.
        """
        if not HAS_MATPLOTLIB:
            return

        models = list(summaries.keys())
        failure_types = ['repetition_loops', 'distractor_contamination', 'format_continuation']

        data = np.zeros((len(models), len(failure_types)))
        for i, model in enumerate(models):
            for j, ftype in enumerate(failure_types):
                data[i, j] = summaries[model].get(ftype, 0)

        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        width = 0.25

        for j, ftype in enumerate(failure_types):
            offset = (j - 1) * width
            ax.bar(x + offset, data[:, j], width,
                   label=ftype.replace('_', ' ').title())

        ax.set_xlabel('Model')
        ax.set_ylabel('Count')
        ax.set_title('Failure Mode Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    def plot_pareto_curve(
        self,
        model_metrics: dict[str, dict[str, float]],
        x_metric: str = "perplexity",
        y_metric: str = "behavioral_accuracy",
        x_label: str = "Perplexity (↓ better)",
        y_label: str = "Behavioral Accuracy (↑ better)",
        title: str = "Perplexity vs Behavioral Accuracy Tradeoff",
        highlight_pareto: bool = True,
        annotate_models: bool = True,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot Pareto curve showing tradeoff between perplexity and behavioral accuracy.

        The Pareto frontier highlights models that are not dominated by any other
        model on both axes simultaneously.

        Args:
            model_metrics: Dict mapping model_id to dict with x_metric and y_metric values
                Example: {
                    'baseline': {'perplexity': 15.2, 'behavioral_accuracy': 0.85},
                    'dba_8_32': {'perplexity': 16.1, 'behavioral_accuracy': 0.82},
                }
            x_metric: Key for x-axis metric (lower is better, e.g., perplexity)
            y_metric: Key for y-axis metric (higher is better, e.g., accuracy)
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            highlight_pareto: Whether to highlight the Pareto frontier
            annotate_models: Whether to add model name labels
            output_path: Where to save the figure
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available for visualization")
            return

        # Extract data points
        models = list(model_metrics.keys())
        x_values = [model_metrics[m].get(x_metric, 0) for m in models]
        y_values = [model_metrics[m].get(y_metric, 0) for m in models]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Find Pareto-optimal points
        # A point is Pareto-optimal if no other point has both lower x AND higher y
        pareto_mask = self._find_pareto_frontier(x_values, y_values)

        # Plot all points
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        for i, (model, x, y) in enumerate(zip(models, x_values, y_values)):
            marker = 's' if pareto_mask[i] else 'o'
            size = 150 if pareto_mask[i] else 100
            edgecolor = 'gold' if pareto_mask[i] else 'black'
            linewidth = 3 if pareto_mask[i] else 1

            ax.scatter(
                x, y,
                c=[colors[i]],
                s=size,
                marker=marker,
                edgecolors=edgecolor,
                linewidths=linewidth,
                label=model,
                zorder=10 if pareto_mask[i] else 5,
            )

            # Add model name annotation
            if annotate_models:
                offset = (5, 5) if not pareto_mask[i] else (8, 8)
                ax.annotate(
                    model,
                    (x, y),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold' if pareto_mask[i] else 'normal',
                )

        # Draw Pareto frontier line
        if highlight_pareto and sum(pareto_mask) > 1:
            pareto_points = [(x_values[i], y_values[i]) for i in range(len(models)) if pareto_mask[i]]
            # Sort by x for line drawing
            pareto_points.sort(key=lambda p: p[0])
            pareto_x, pareto_y = zip(*pareto_points)

            # Draw stepped line connecting Pareto points
            ax.plot(
                pareto_x, pareto_y,
                'g--', linewidth=2, alpha=0.7,
                label='Pareto Frontier',
                zorder=1,
            )

            # Shade the dominated region
            # Everything to the right and below the frontier is dominated
            x_fill = list(pareto_x) + [max(x_values) * 1.1, max(x_values) * 1.1, pareto_x[0]]
            y_fill = list(pareto_y) + [pareto_y[-1], 0, 0]
            ax.fill(x_fill, y_fill, alpha=0.1, color='red', label='Dominated Region')

        # Formatting
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Legend
        ax.legend(loc='lower left', fontsize=9)

        # Add annotation explaining the plot
        ax.text(
            0.98, 0.02,
            "★ = Pareto-optimal (not dominated)",
            transform=ax.transAxes,
            fontsize=8,
            ha='right',
            va='bottom',
            style='italic',
            alpha=0.7,
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _find_pareto_frontier(
        self,
        x_values: list[float],
        y_values: list[float],
    ) -> list[bool]:
        """
        Find Pareto-optimal points.

        A point is Pareto-optimal if no other point has both:
        - Lower x value (better perplexity)
        - Higher y value (better accuracy)

        Args:
            x_values: X-axis values (lower is better)
            y_values: Y-axis values (higher is better)

        Returns:
            Boolean mask indicating which points are Pareto-optimal
        """
        n = len(x_values)
        pareto_mask = [True] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if point j dominates point i
                # j dominates i if j has lower x AND higher y
                if x_values[j] <= x_values[i] and y_values[j] >= y_values[i]:
                    # j is at least as good on both axes
                    if x_values[j] < x_values[i] or y_values[j] > y_values[i]:
                        # j is strictly better on at least one axis
                        pareto_mask[i] = False
                        break

        return pareto_mask

    def plot_pareto_from_results(
        self,
        summaries: dict[str, dict[str, Any]],
        perplexities: dict[str, float],
        accuracy_metric: str = "exact_match_rate",
        output_path: Path | None = None,
    ) -> None:
        """
        Convenience method to plot Pareto curve directly from evaluation results.

        Args:
            summaries: Model summaries from EvalResults.summaries
            perplexities: Dict mapping model_id to perplexity value
            accuracy_metric: Which accuracy metric to use from summaries
            output_path: Where to save
        """
        model_metrics = {}
        for model_id, summary in summaries.items():
            if model_id in perplexities:
                model_metrics[model_id] = {
                    'perplexity': perplexities[model_id],
                    'behavioral_accuracy': summary.get(accuracy_metric, 0),
                }

        if not model_metrics:
            print("Warning: No models with both perplexity and accuracy data")
            return

        self.plot_pareto_curve(
            model_metrics=model_metrics,
            output_path=output_path,
        )


def generate_html_report(
    results: Any,  # EvalResults
    output_path: Path,
) -> None:
    """
    Generate interactive HTML dashboard.
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DBA Behavioral Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary-card {{
            background: #f5f5f5;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            display: inline-block;
            min-width: 200px;
        }}
        .metric {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .label {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .winner {{ background-color: #c8e6c9; }}
        .loser {{ background-color: #ffcdd2; }}
        h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .score-3 {{ background-color: #4CAF50; color: white; }}
        .score-2 {{ background-color: #8BC34A; }}
        .score-1 {{ background-color: #FFEB3B; }}
        .score-0 {{ background-color: #FF9800; color: white; }}
        .score--1 {{ background-color: #f44336; color: white; }}
    </style>
</head>
<body>
    <h1>DBA Behavioral Evaluation Report</h1>
    <p>Generated: {timestamp}</p>

    <h2>Summary</h2>
    <div class="summary-cards">
        {summary_cards}
    </div>

    <h2>Head-to-Head Comparisons</h2>
    <table>
        <tr>
            <th>Model A</th>
            <th>Model B</th>
            <th>Wins A</th>
            <th>Wins B</th>
            <th>Ties</th>
        </tr>
        {comparison_rows}
    </table>

    <h2>Per-Category Results</h2>
    {category_tables}

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Test ID</th>
            {model_headers}
        </tr>
        {detail_rows}
    </table>
</body>
</html>
"""

    from datetime import datetime

    # Generate summary cards
    summary_cards = ""
    for model_id, summary in results.summaries.items():
        summary_cards += f"""
        <div class="summary-card">
            <div class="label">{model_id}</div>
            <div class="metric">{summary['exact_match_rate']:.1%}</div>
            <div class="label">Exact Match Rate</div>
            <div class="metric">{summary['soft_score_avg']:.2f}</div>
            <div class="label">Avg Soft Score</div>
        </div>
        """

    # Generate comparison rows
    comparison_rows = ""
    for comp in results.comparisons:
        comparison_rows += f"""
        <tr>
            <td>{comp['model_a']}</td>
            <td>{comp['model_b']}</td>
            <td class="{'winner' if comp['wins_a'] > comp['wins_b'] else ''}">{comp['wins_a']}</td>
            <td class="{'winner' if comp['wins_b'] > comp['wins_a'] else ''}">{comp['wins_b']}</td>
            <td>{comp['ties']}</td>
        </tr>
        """

    # Generate category tables
    category_tables = ""
    for category, model_results in results.category_results.items():
        category_tables += f"<h3>{category}</h3><table>"
        category_tables += "<tr><th>Model</th><th>Exact Match</th><th>Content Match</th><th>Avg Score</th></tr>"
        for model_id, stats in model_results.items():
            category_tables += f"""
            <tr>
                <td>{model_id}</td>
                <td>{stats['exact_match_rate']:.1%}</td>
                <td>{stats['content_match_rate']:.1%}</td>
                <td>{stats['soft_score_avg']:.2f}</td>
            </tr>
            """
        category_tables += "</table>"

    # Generate model headers
    model_headers = "".join(f"<th>{mid}</th>" for mid in results.model_ids)

    # Generate detail rows (sample first 50)
    detail_rows = ""
    test_ids = list(results.scores[results.model_ids[0]].keys())[:50]
    for test_id in test_ids:
        detail_rows += f"<tr><td>{test_id}</td>"
        for model_id in results.model_ids:
            score = results.scores[model_id].get(test_id)
            if score:
                score_class = f"score-{score.soft_score.value}"
                detail_rows += f'<td class="{score_class}">{score.soft_score.name}</td>'
            else:
                detail_rows += "<td>-</td>"
        detail_rows += "</tr>"

    # Render template
    html = html_template.format(
        timestamp=datetime.now().isoformat(),
        summary_cards=summary_cards,
        comparison_rows=comparison_rows,
        category_tables=category_tables,
        model_headers=model_headers,
        detail_rows=detail_rows,
    )

    output_path.write_text(html)
    print(f"HTML report saved to: {output_path}")
