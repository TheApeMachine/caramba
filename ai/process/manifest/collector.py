"""Results collector for gathering experiment outputs.

Provides utilities for AI agents to collect, summarize, and analyze
results from completed experiment runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from caramba.console import logger


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    name: str
    timestamp: datetime
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    error: str | None = None


@dataclass
class ResultsSummary:
    """Summary of results from multiple experiments."""

    experiments: list[ExperimentResult]
    best_by_metric: dict[str, ExperimentResult] = field(default_factory=dict)
    comparisons: list[dict[str, Any]] = field(default_factory=list)


class ResultsCollector:
    """Collector for gathering and summarizing experiment results.

    Scans the artifacts directory for experiment outputs and provides
    methods to aggregate, compare, and summarize results.
    """

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)

    def collect_experiment_results(self, experiment_name: str | None = None) -> list[ExperimentResult]:
        """Collect results from experiment runs.

        Args:
            experiment_name: Optional filter for specific experiment

        Returns:
            List of ExperimentResult objects
        """
        results: list[ExperimentResult] = []

        if not self.artifacts_dir.exists():
            logger.warning(f"Artifacts directory not found: {self.artifacts_dir}")
            return results

        # Look for experiment directories
        for exp_dir in self.artifacts_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            if experiment_name and exp_dir.name != experiment_name:
                continue

            result = self._parse_experiment_dir(exp_dir)
            if result:
                results.append(result)

        return sorted(results, key=lambda r: r.timestamp, reverse=True)

    def _parse_experiment_dir(self, exp_dir: Path) -> ExperimentResult | None:
        """Parse a single experiment directory for results."""
        try:
            # Look for metrics file
            metrics_path = exp_dir / "metrics.json"
            metrics = {}
            if metrics_path.exists():
                with metrics_path.open("r") as f:
                    metrics = json.load(f)

            # Look for config file
            config_path = exp_dir / "config.json"
            config = {}
            if config_path.exists():
                with config_path.open("r") as f:
                    config = json.load(f)

            # Collect artifact files
            artifacts = list(exp_dir.glob("**/*"))
            artifacts = [p for p in artifacts if p.is_file()]

            # Get timestamp from directory or most recent file
            try:
                timestamp = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            except Exception:
                timestamp = datetime.now()

            # Determine status
            status = "completed"
            error = None
            error_path = exp_dir / "error.txt"
            if error_path.exists():
                status = "failed"
                error = error_path.read_text(encoding="utf-8").strip()

            return ExperimentResult(
                name=exp_dir.name,
                timestamp=timestamp,
                metrics=metrics,
                artifacts=artifacts,
                config=config,
                status=status,
                error=error,
            )

        except Exception as e:
            logger.warning(f"Failed to parse experiment directory {exp_dir}: {e}")
            return None

    def summarize_results(self, results: list[ExperimentResult]) -> ResultsSummary:
        """Create a summary of experiment results.

        Identifies best performers by each metric and creates comparisons.
        """
        summary = ResultsSummary(experiments=results)

        if not results:
            return summary

        # Find best by each metric (assuming lower is better for loss-type metrics)
        metric_names: set[str] = set()
        for result in results:
            metric_names.update(result.metrics.keys())

        for metric in metric_names:
            results_with_metric = [r for r in results if metric in r.metrics]
            if not results_with_metric:
                continue

            # Determine if lower is better (heuristic based on metric name)
            lower_is_better = any(
                name in metric.lower() for name in ["loss", "error", "perplexity"]
            )

            if lower_is_better:
                best = min(results_with_metric, key=lambda r: r.metrics[metric])
            else:
                best = max(results_with_metric, key=lambda r: r.metrics[metric])

            summary.best_by_metric[metric] = best

        # Create pairwise comparisons for the most recent experiments
        recent = sorted(results, key=lambda r: r.timestamp, reverse=True)[:5]
        for i, r1 in enumerate(recent):
            for r2 in recent[i + 1 :]:
                comparison = self._compare_experiments(r1, r2)
                if comparison:
                    summary.comparisons.append(comparison)

        return summary

    def _compare_experiments(
        self, exp1: ExperimentResult, exp2: ExperimentResult
    ) -> dict[str, Any] | None:
        """Compare two experiments."""
        common_metrics = set(exp1.metrics.keys()) & set(exp2.metrics.keys())
        if not common_metrics:
            return None

        comparison = {
            "experiment_1": exp1.name,
            "experiment_2": exp2.name,
            "metrics": {},
        }

        for metric in common_metrics:
            v1 = exp1.metrics[metric]
            v2 = exp2.metrics[metric]
            if v2 != 0:
                diff_pct = ((v1 - v2) / abs(v2)) * 100
            else:
                diff_pct = 0.0

            comparison["metrics"][metric] = {
                "exp1_value": v1,
                "exp2_value": v2,
                "difference": v1 - v2,
                "difference_pct": diff_pct,
            }

        return comparison

    def format_summary_markdown(self, summary: ResultsSummary) -> str:
        """Format a results summary as Markdown."""
        lines = ["# Experiment Results Summary\n"]

        if not summary.experiments:
            lines.append("No experiments found.\n")
            return "\n".join(lines)

        lines.append(f"**Total experiments:** {len(summary.experiments)}\n")

        # Best by metric
        if summary.best_by_metric:
            lines.append("## Best Results by Metric\n")
            for metric, result in summary.best_by_metric.items():
                value = result.metrics.get(metric, "N/A")
                lines.append(f"- **{metric}:** {value:.4f} ({result.name})")
            lines.append("")

        # Recent experiments
        lines.append("## Recent Experiments\n")
        recent = sorted(summary.experiments, key=lambda r: r.timestamp, reverse=True)[:10]
        for exp in recent:
            status_emoji = "Done" if exp.status == "completed" else "Failed"
            lines.append(f"### {exp.name} [{status_emoji}]\n")
            lines.append(f"- **Timestamp:** {exp.timestamp.isoformat()}")
            if exp.metrics:
                lines.append("- **Metrics:**")
                for k, v in exp.metrics.items():
                    lines.append(f"  - {k}: {v:.4f}")
            lines.append("")

        return "\n".join(lines)

    def get_latest_metrics(self, metric_name: str) -> list[tuple[str, float, datetime]]:
        """Get the latest value of a specific metric across all experiments.

        Returns:
            List of (experiment_name, metric_value, timestamp) tuples
        """
        results = self.collect_experiment_results()
        metric_values = []

        for result in results:
            if metric_name in result.metrics:
                metric_values.append(
                    (result.name, result.metrics[metric_name], result.timestamp)
                )

        return sorted(metric_values, key=lambda x: x[2], reverse=True)
