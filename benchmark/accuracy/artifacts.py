"""
Rich artifact generation for accuracy benchmarks.

Produces comprehensive outputs matching the behavioral suite:
- JSON: summary and detailed per-sample results
- CSV: tabular summaries for easy analysis
- LaTeX: publication-ready tables
- Markdown: human-readable reports
- Visualizations: comparison charts and heatmaps
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from collector.measurement.accuracy.result import AccuracyResult
from collector.measurement.accuracy.task import TaskAccuracy


@dataclass
class AccuracyArtifactConfig:
    """Configuration for artifact generation."""
    save_json_summary: bool = True
    save_json_detailed: bool = True
    save_csv: bool = True
    save_latex: bool = True
    save_markdown: bool = True
    save_visualizations: bool = True


@dataclass
class MultiModelAccuracyResults:
    """Aggregated results from multiple models for comparison."""
    model_results: dict[str, AccuracyResult] = field(default_factory=dict)
    baseline_name: str | None = None

    def get_task_accuracy(self, model_name: str, task_name: str) -> float:
        """Get accuracy for a specific model and task."""
        result = self.model_results.get(model_name)
        if result:
            for task in result.tasks:
                if task.task == task_name:
                    return task.accuracy
        return 0.0

    def get_all_task_names(self) -> list[str]:
        """Get union of all task names across models."""
        tasks = set()
        for result in self.model_results.values():
            for task in result.tasks:
                tasks.add(task.task)
        return sorted(tasks)

    def get_model_names(self) -> list[str]:
        """Get all model names."""
        return list(self.model_results.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "baseline_name": self.baseline_name,
            "model_results": {
                name: {
                    "model_name": result.model_name,
                    "micro_accuracy": result.micro_accuracy,
                    "tasks": [
                        {
                            "task": t.task,
                            "split": t.split,
                            "accuracy": t.accuracy,
                            "correct": t.correct,
                            "total": t.total,
                        }
                        for t in result.tasks
                    ],
                }
                for name, result in self.model_results.items()
            },
        }


class AccuracyArtifactGenerator:
    """
    Generate rich artifacts for accuracy benchmark results.

    Produces outputs matching behavioral suite quality:
    - accuracy_summary.json: High-level metrics
    - accuracy_detailed.json: Per-sample results
    - accuracy_summary.csv: Tabular data
    - accuracy_table.tex: LaTeX table
    - accuracy_report.md: Markdown report
    - accuracy_comparison.png: Visualization (if matplotlib available)
    """

    def __init__(self, config: AccuracyArtifactConfig | None = None):
        self.config = config or AccuracyArtifactConfig()

    def generate_all(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str = "accuracy_",
    ) -> dict[str, Path]:
        """Generate all configured artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        if self.config.save_json_summary:
            path = self._save_json_summary(results, output_dir, prefix)
            paths["json_summary"] = path

        if self.config.save_json_detailed:
            path = self._save_json_detailed(results, output_dir, prefix)
            paths["json_detailed"] = path

        if self.config.save_csv:
            path = self._save_csv(results, output_dir, prefix)
            paths["csv"] = path

        if self.config.save_latex:
            path = self._save_latex(results, output_dir, prefix)
            paths["latex"] = path

        if self.config.save_markdown:
            path = self._save_markdown(results, output_dir, prefix)
            paths["markdown"] = path

        if self.config.save_visualizations:
            viz_paths = self._save_visualizations(results, output_dir, prefix)
            paths.update(viz_paths)

        return paths

    def _save_json_summary(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save high-level summary JSON."""
        path = output_dir / f"{prefix}summary.json"

        summary = {
            "baseline": results.baseline_name,
            "models": {},
            "tasks": results.get_all_task_names(),
        }

        for model_name, result in results.model_results.items():
            summary["models"][model_name] = {
                "micro_accuracy": result.micro_accuracy,
                "task_count": len(result.tasks),
                "total_examples": sum(t.total for t in result.tasks),
                "total_correct": sum(t.correct for t in result.tasks),
                "per_task": {t.task: t.accuracy for t in result.tasks},
            }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path

    def _save_json_detailed(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save detailed per-sample JSON."""
        path = output_dir / f"{prefix}detailed.json"

        detailed = {
            "baseline": results.baseline_name,
            "models": {},
        }

        for model_name, result in results.model_results.items():
            model_data = {
                "micro_accuracy": result.micro_accuracy,
                "tasks": {},
            }

            for task in result.tasks:
                model_data["tasks"][task.task] = {
                    "split": task.split,
                    "accuracy": task.accuracy,
                    "correct": task.correct,
                    "total": task.total,
                    "samples": [
                        {
                            "prompt": s.prompt[:500] + "..." if len(s.prompt) > 500 else s.prompt,
                            "gold": s.gold,
                            "pred": s.pred,
                            "ok": s.ok,
                        }
                        for s in (task.samples or [])
                    ],
                }

            detailed["models"][model_name] = model_data

        with open(path, "w") as f:
            json.dump(detailed, f, indent=2)

        return path

    def _save_csv(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save CSV summary table."""
        path = output_dir / f"{prefix}summary.csv"

        task_names = results.get_all_task_names()
        model_names = results.get_model_names()

        with open(path, "w") as f:
            # Header
            headers = ["model", "micro_accuracy"] + task_names
            f.write(",".join(headers) + "\n")

            # Rows
            for model_name in model_names:
                result = results.model_results[model_name]
                row = [
                    model_name,
                    f"{result.micro_accuracy:.4f}",
                ]
                for task_name in task_names:
                    acc = results.get_task_accuracy(model_name, task_name)
                    row.append(f"{acc:.4f}")
                f.write(",".join(row) + "\n")

        return path

    def _save_latex(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save LaTeX table."""
        path = output_dir / f"{prefix}table.tex"

        task_names = results.get_all_task_names()
        model_names = results.get_model_names()

        # Find best accuracy for each task to bold
        best_per_task = {}
        for task_name in task_names:
            best_acc = -1
            best_model = None
            for model_name in model_names:
                acc = results.get_task_accuracy(model_name, task_name)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name
            best_per_task[task_name] = best_model

        # Find best micro accuracy
        best_micro_model = max(
            model_names,
            key=lambda m: results.model_results[m].micro_accuracy,
        )

        with open(path, "w") as f:
            f.write("% Auto-generated LaTeX table for accuracy benchmark results\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Accuracy Benchmark Results}\n")
            f.write("\\label{tab:accuracy}\n")

            # Column specification
            col_spec = "l" + "r" * (len(task_names) + 1)
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write("\\toprule\n")

            # Header
            task_headers = [t.replace("_", "\\_") for t in task_names]
            f.write("Model & Micro & " + " & ".join(task_headers) + " \\\\\n")
            f.write("\\midrule\n")

            # Rows
            for model_name in model_names:
                result = results.model_results[model_name]
                escaped_name = model_name.replace("_", "\\_")

                # Micro accuracy
                micro_str = f"{result.micro_accuracy * 100:.1f}\\%"
                if model_name == best_micro_model:
                    micro_str = f"\\textbf{{{micro_str}}}"

                cells = [escaped_name, micro_str]

                for task_name in task_names:
                    acc = results.get_task_accuracy(model_name, task_name)
                    acc_str = f"{acc * 100:.1f}\\%"
                    if best_per_task.get(task_name) == model_name:
                        acc_str = f"\\textbf{{{acc_str}}}"
                    cells.append(acc_str)

                f.write(" & ".join(cells) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        return path

    def _save_markdown(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save markdown report."""
        path = output_dir / f"{prefix}report.md"

        task_names = results.get_all_task_names()
        model_names = results.get_model_names()

        with open(path, "w") as f:
            f.write("# Accuracy Benchmark Results\n\n")

            if results.baseline_name:
                f.write(f"**Baseline:** {results.baseline_name}\n\n")

            f.write(f"**Models:** {', '.join(model_names)}\n\n")
            f.write(f"**Tasks:** {', '.join(task_names)}\n\n")

            # Summary table
            f.write("## Summary\n\n")
            header = "| Model | Micro Acc | " + " | ".join(task_names) + " |\n"
            separator = "|-------|-----------|" + "|".join(["-----------"] * len(task_names)) + "|\n"
            f.write(header)
            f.write(separator)

            for model_name in model_names:
                result = results.model_results[model_name]
                row = f"| {model_name} | {result.micro_accuracy * 100:.1f}% |"
                for task_name in task_names:
                    acc = results.get_task_accuracy(model_name, task_name)
                    row += f" {acc * 100:.1f}% |"
                f.write(row + "\n")

            f.write("\n")

            # Per-task details
            f.write("## Per-Task Details\n\n")
            for task_name in task_names:
                f.write(f"### {task_name}\n\n")
                f.write("| Model | Correct | Total | Accuracy |\n")
                f.write("|-------|---------|-------|----------|\n")

                for model_name in model_names:
                    result = results.model_results[model_name]
                    for task in result.tasks:
                        if task.task == task_name:
                            f.write(
                                f"| {model_name} | {task.correct} | {task.total} | "
                                f"{task.accuracy * 100:.1f}% |\n"
                            )

                f.write("\n")

            # Delta from baseline if available
            if results.baseline_name and results.baseline_name in model_names:
                f.write("## Delta from Baseline\n\n")
                baseline = results.model_results[results.baseline_name]

                f.write("| Model | Micro Δ | " + " | ".join(f"{t} Δ" for t in task_names) + " |\n")
                f.write("|-------|---------|" + "|".join(["--------"] * len(task_names)) + "|\n")

                for model_name in model_names:
                    if model_name == results.baseline_name:
                        continue
                    result = results.model_results[model_name]

                    micro_delta = (result.micro_accuracy - baseline.micro_accuracy) * 100
                    sign = "+" if micro_delta >= 0 else ""
                    row = f"| {model_name} | {sign}{micro_delta:.1f}% |"

                    for task_name in task_names:
                        model_acc = results.get_task_accuracy(model_name, task_name)
                        baseline_acc = results.get_task_accuracy(results.baseline_name, task_name)
                        delta = (model_acc - baseline_acc) * 100
                        sign = "+" if delta >= 0 else ""
                        row += f" {sign}{delta:.1f}% |"

                    f.write(row + "\n")

                f.write("\n")

        return path

    def _save_visualizations(
        self,
        results: MultiModelAccuracyResults,
        output_dir: Path,
        prefix: str,
    ) -> dict[str, Path]:
        """Save visualization charts."""
        paths = {}

        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np
        except ImportError:
            return paths

        task_names = results.get_all_task_names()
        model_names = results.get_model_names()

        # 1. Bar chart: accuracy by task
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(task_names))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            accuracies = [
                results.get_task_accuracy(model_name, task_name)
                for task_name in task_names
            ]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, [a * 100 for a in accuracies], width, label=model_name)

        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy (%)")
        # No title (paper-friendly; caption covers this).
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"{prefix}by_task.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        paths["viz_by_task"] = path

        # 2. Heatmap: model x task
        if len(model_names) > 1:
            fig, ax = plt.subplots(figsize=(10, max(4, len(model_names))))

            data = np.array([
                [results.get_task_accuracy(m, t) * 100 for t in task_names]
                for m in model_names
            ])

            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

            ax.set_xticks(np.arange(len(task_names)))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(task_names, rotation=45, ha="right")
            ax.set_yticklabels(model_names)

            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(task_names)):
                    text = ax.text(j, i, f"{data[i, j]:.1f}",
                                   ha="center", va="center", color="black", fontsize=8)

            # No title (paper-friendly).
            cbar = plt.colorbar(im)
            cbar.set_label("Accuracy (%)")

            plt.tight_layout()
            path = output_dir / f"{prefix}heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            paths["viz_heatmap"] = path

        # 3. Micro accuracy comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 5))

        micro_accs = [
            results.model_results[m].micro_accuracy * 100
            for m in model_names
        ]

        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, micro_accs, color=colors)

        # Add value labels
        for bar, acc in zip(bars, micro_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Accuracy (%)")
        # No title (paper-friendly).
        ax.set_ylim(0, max(micro_accs) * 1.15)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"{prefix}micro_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        paths["viz_micro"] = path

        return paths


def generate_accuracy_artifacts(
    results: dict[str, AccuracyResult] | MultiModelAccuracyResults,
    output_dir: Path,
    baseline_name: str | None = None,
    prefix: str = "accuracy_",
    config: AccuracyArtifactConfig | None = None,
) -> dict[str, Path]:
    """
    Convenience function to generate all accuracy artifacts.

    Args:
        results: Either a dict of model_name -> AccuracyResult, or MultiModelAccuracyResults
        output_dir: Directory to save artifacts
        baseline_name: Name of baseline model for delta calculations
        prefix: Prefix for artifact filenames
        config: Artifact generation config

    Returns:
        Dict mapping artifact names to their file paths
    """
    if isinstance(results, dict):
        multi_results = MultiModelAccuracyResults(
            model_results=results,
            baseline_name=baseline_name,
        )
    else:
        multi_results = results

    generator = AccuracyArtifactGenerator(config)
    return generator.generate_all(multi_results, output_dir, prefix)
