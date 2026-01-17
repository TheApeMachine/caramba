"""Multi-model artifact generation for N-way comparisons.

Generates paper-ready artifacts (CSV, JSON, PNG, LaTeX) for comparing
N models across multiple benchmarks with comprehensive metrics,
rankings, and visualizations.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from caramba.benchmark.accuracy import AccuracyResult
from caramba.benchmark.artifacts import ExperimentMetadata
from caramba.benchmark.context import ContextResult
from caramba.benchmark.latency import LatencyResult
from caramba.benchmark.memory import MemoryResult
from caramba.benchmark.multi_model_results import (
    MultiModelComparisonSummary,
    build_comparison_summary,
)
from caramba.benchmark.perplexity import PerplexityResult
from caramba.console import logger

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# Color scheme for N models (up to 10)
MODEL_COLORS = [
    "#2ecc71",  # green (baseline)
    "#3498db",  # blue
    "#e74c3c",  # red
    "#9b59b6",  # purple
    "#f39c12",  # orange
    "#1abc9c",  # teal
    "#e91e63",  # pink
    "#00bcd4",  # cyan
    "#ff5722",  # deep orange
    "#795548",  # brown
]


def get_model_color(model_name: str, model_names: list[str]) -> str:
    """Get consistent color for a model."""
    if model_name in model_names:
        idx = model_names.index(model_name) % len(MODEL_COLORS)
        return MODEL_COLORS[idx]
    return MODEL_COLORS[0]


class MultiModelArtifactGenerator:
    """Generates paper-ready artifacts for N-model comparisons."""

    def __init__(
        self,
        output_dir: Path | str,
        baseline_name: str | None = None,
    ) -> None:
        """Initialize the artifact generator.

        Args:
            output_dir: Directory to write artifacts to.
            baseline_name: Name of baseline model for delta calculations.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_name = baseline_name

    def generate_all(
        self,
        *,
        metadata: ExperimentMetadata,
        perplexity_results: dict[str, PerplexityResult] | None = None,
        latency_results: dict[str, LatencyResult] | None = None,
        memory_results: dict[str, MemoryResult] | None = None,
        accuracy_results: dict[str, AccuracyResult] | None = None,
        context_results: dict[str, ContextResult] | None = None,
        behavioral_results: dict[str, dict] | None = None,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate all multi-model artifacts.

        Args:
            metadata: Experiment metadata.
            perplexity_results: Dict mapping model names to PerplexityResult.
            latency_results: Dict mapping model names to LatencyResult.
            memory_results: Dict mapping model names to MemoryResult.
            accuracy_results: Dict mapping model names to AccuracyResult.
            context_results: Dict mapping model names to ContextResult.
            behavioral_results: Dict mapping model names to behavioral summary dicts.
            formats: List of output formats ("csv", "json", "png", "latex").

        Returns:
            Dict mapping artifact names to file paths.
        """
        formats = formats or ["csv", "json", "png", "latex"]
        generated: dict[str, Path] = {}

        # Build comprehensive summary
        summary = build_comparison_summary(
            perplexity_results=perplexity_results,
            latency_results=latency_results,
            memory_results=memory_results,
            accuracy_results=accuracy_results,
            context_results=context_results,
            behavioral_results=behavioral_results,
            baseline_name=self.baseline_name,
        )

        if "json" in formats:
            paths = self._write_json_report(
                metadata, summary, perplexity_results, latency_results,
                memory_results, accuracy_results, context_results, behavioral_results
            )
            generated.update(paths)

        if "csv" in formats:
            paths = self._write_csv_files(
                summary, perplexity_results, latency_results,
                memory_results, accuracy_results, context_results
            )
            generated.update(paths)

        if "png" in formats and HAS_MATPLOTLIB:
            paths = self._generate_charts(
                summary, perplexity_results, latency_results,
                memory_results, accuracy_results, context_results, behavioral_results
            )
            generated.update(paths)
        elif "png" in formats:
            logger.warning("matplotlib not available, skipping PNG generation")

        if "latex" in formats:
            paths = self._write_latex_tables(
                metadata, summary,
                perplexity_results=perplexity_results,
                latency_results=latency_results,
                memory_results=memory_results,
                accuracy_results=accuracy_results,
                context_results=context_results,
            )
            generated.update(paths)

        return generated

    def _write_json_report(
        self,
        metadata: ExperimentMetadata,
        summary: MultiModelComparisonSummary,
        perplexity_results: dict[str, PerplexityResult] | None,
        latency_results: dict[str, LatencyResult] | None,
        memory_results: dict[str, MemoryResult] | None,
        accuracy_results: dict[str, AccuracyResult] | None,
        context_results: dict[str, ContextResult] | None,
        behavioral_results: dict[str, dict] | None,
    ) -> dict[str, Path]:
        """Write comprehensive JSON report."""
        report = {
            "metadata": {
                "name": metadata.name,
                "timestamp": metadata.timestamp,
                "manifest_path": metadata.manifest_path,
                "device": metadata.device,
                "notes": metadata.notes,
                "generated_at": datetime.now().isoformat(),
            },
            "summary": summary.to_dict(),
            "detailed_results": {},
        }

        # Add detailed per-model results
        if perplexity_results:
            report["detailed_results"]["perplexity"] = {
                name: {
                    "perplexity": r.perplexity,
                    "loss": r.loss,
                    "num_tokens": r.num_tokens,
                    "num_batches": r.num_batches,
                }
                for name, r in perplexity_results.items()
            }

        if latency_results:
            report["detailed_results"]["latency"] = {
                name: {
                    "avg_tokens_per_second": r.avg_tokens_per_second,
                    "avg_time_to_first_token_ms": r.avg_time_to_first_token_ms,
                    "measurements": [
                        {
                            "prompt_len": m.prompt_len,
                            "gen_len": m.gen_len,
                            "batch_size": m.batch_size,
                            "prefill_time_ms": m.prefill_time_ms,
                            "decode_time_ms": m.decode_time_ms,
                            "tokens_per_second": m.tokens_per_second,
                            "time_to_first_token_ms": m.time_to_first_token_ms,
                        }
                        for m in r.measurements
                    ],
                }
                for name, r in latency_results.items()
            }

        if memory_results:
            report["detailed_results"]["memory"] = {
                name: {
                    "peak_memory_mb": r.peak_memory_mb,
                    "kvcache_analysis": (
                        {
                            "n_layers": r.kvcache_analysis.n_layers,
                            "n_kv_heads": r.kvcache_analysis.n_kv_heads,
                            "head_dim": r.kvcache_analysis.head_dim,
                            "attention_mode": r.kvcache_analysis.attention_mode,
                            "bytes_per_token_fp16": r.kvcache_analysis.bytes_per_token_fp16,
                            "bytes_per_token_dba_fp16": r.kvcache_analysis.bytes_per_token_dba_fp16,
                        }
                        if r.kvcache_analysis
                        else None
                    ),
                    "measurements": [
                        {
                            "seq_len": m.seq_len,
                            "batch_size": m.batch_size,
                            "peak_memory_mb": m.peak_memory_mb,
                            "kvcache_memory_mb": m.kvcache_memory_mb,
                        }
                        for m in r.measurements
                    ],
                }
                for name, r in memory_results.items()
            }

        if accuracy_results:
            report["detailed_results"]["accuracy"] = {
                name: {
                    "micro_accuracy": r.micro_accuracy,
                    "tasks": [
                        {
                            "task": t.task,
                            "split": t.split,
                            "accuracy": t.accuracy,
                            "correct": t.correct,
                            "total": t.total,
                        }
                        for t in r.tasks
                    ],
                }
                for name, r in accuracy_results.items()
            }

        if behavioral_results:
            report["detailed_results"]["behavioral"] = behavioral_results

        path = self.output_dir / "report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return {"report.json": path}

    def _write_csv_files(
        self,
        summary: MultiModelComparisonSummary,
        perplexity_results: dict[str, PerplexityResult] | None,
        latency_results: dict[str, LatencyResult] | None,
        memory_results: dict[str, MemoryResult] | None,
        accuracy_results: dict[str, AccuracyResult] | None,
        context_results: dict[str, ContextResult] | None,
    ) -> dict[str, Path]:
        """Write CSV files for all metrics."""
        generated: dict[str, Path] = {}

        # Summary CSV
        path = self.output_dir / "comparison_summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "perplexity", "tokens_per_sec", "kv_bytes_per_token",
                "ppl_rank", "throughput_rank", "memory_rank",
                "ppl_delta", "speedup", "memory_reduction",
                "efficiency_score", "pareto_optimal",
            ])
            for name in summary.model_names:
                writer.writerow([
                    name,
                    summary.perplexities.get(name, ""),
                    summary.throughputs.get(name, ""),
                    summary.kv_bytes_per_token.get(name, ""),
                    summary.get_rank(name, "perplexity") or "",
                    summary.get_rank(name, "throughput") or "",
                    summary.get_rank(name, "memory") or "",
                    (summary.perplexity_deltas or {}).get(name, ""),
                    (summary.speedups or {}).get(name, ""),
                    (summary.memory_reductions or {}).get(name, ""),
                    summary.efficiency_scores.get(name, ""),
                    "yes" if name in summary.pareto_optimal else "no",
                ])
        generated["comparison_summary.csv"] = path

        # Detailed perplexity CSV
        if perplexity_results:
            path = self.output_dir / "perplexity.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "perplexity", "loss", "num_tokens", "num_batches"])
                for name, r in perplexity_results.items():
                    writer.writerow([name, r.perplexity, r.loss, r.num_tokens, r.num_batches])
            generated["perplexity.csv"] = path

        # Detailed latency CSV
        if latency_results:
            path = self.output_dir / "latency.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "prompt_len", "gen_len", "batch_size",
                    "prefill_time_ms", "decode_time_ms", "tokens_per_second",
                    "time_to_first_token_ms",
                ])
                for name, r in latency_results.items():
                    for m in r.measurements:
                        writer.writerow([
                            name, m.prompt_len, m.gen_len, m.batch_size,
                            m.prefill_time_ms, m.decode_time_ms,
                            m.tokens_per_second, m.time_to_first_token_ms,
                        ])
            generated["latency.csv"] = path

        # Detailed memory CSV
        if memory_results:
            path = self.output_dir / "memory.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "seq_len", "batch_size",
                    "peak_memory_mb", "kvcache_memory_mb",
                    "kv_bytes_per_token_fp16", "kv_bytes_per_token_dba_fp16",
                ])
                for name, r in memory_results.items():
                    kv_fp16 = r.kvcache_analysis.bytes_per_token_fp16 if r.kvcache_analysis else ""
                    kv_dba = r.kvcache_analysis.bytes_per_token_dba_fp16 if r.kvcache_analysis else ""
                    for m in r.measurements:
                        writer.writerow([
                            name, m.seq_len, m.batch_size,
                            m.peak_memory_mb, m.kvcache_memory_mb,
                            kv_fp16, kv_dba,
                        ])
            generated["memory.csv"] = path

        # Rankings CSV
        path = self.output_dir / "rankings.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "perplexity", "throughput", "memory", "accuracy", "behavioral"])
            max_len = max(
                len(summary.perplexity_rankings),
                len(summary.throughput_rankings),
                len(summary.memory_rankings),
                len(summary.accuracy_rankings),
                len(summary.behavioral_rankings),
            )
            for i in range(max_len):
                writer.writerow([
                    i + 1,
                    summary.perplexity_rankings[i] if i < len(summary.perplexity_rankings) else "",
                    summary.throughput_rankings[i] if i < len(summary.throughput_rankings) else "",
                    summary.memory_rankings[i] if i < len(summary.memory_rankings) else "",
                    summary.accuracy_rankings[i] if i < len(summary.accuracy_rankings) else "",
                    summary.behavioral_rankings[i] if i < len(summary.behavioral_rankings) else "",
                ])
        generated["rankings.csv"] = path

        return generated

    def _generate_charts(
        self,
        summary: MultiModelComparisonSummary,
        perplexity_results: dict[str, PerplexityResult] | None,
        latency_results: dict[str, LatencyResult] | None,
        memory_results: dict[str, MemoryResult] | None,
        accuracy_results: dict[str, AccuracyResult] | None,
        context_results: dict[str, ContextResult] | None,
        behavioral_results: dict[str, dict] | None,
    ) -> dict[str, Path]:
        """Generate all PNG charts."""
        generated: dict[str, Path] = {}

        model_names = summary.model_names
        colors = [get_model_color(m, model_names) for m in model_names]

        # 1. Summary bar chart (perplexity, throughput, memory)
        if summary.perplexities or summary.throughputs or summary.kv_bytes_per_token:
            path = self._plot_summary_bars(summary, model_names, colors)
            generated["comparison_summary.png"] = path

        # 2. Latency vs context length
        if latency_results:
            path = self._plot_latency_vs_context(latency_results, model_names, colors)
            if path:
                generated["latency_vs_context.png"] = path

        # 3. Memory vs context length
        if memory_results:
            path = self._plot_memory_vs_context(memory_results, model_names, colors)
            if path:
                generated["memory_vs_context.png"] = path

        # 4. Pareto frontier
        if summary.perplexities and summary.kv_bytes_per_token:
            path = self._plot_pareto_frontier(summary, model_names, colors)
            generated["pareto_frontier.png"] = path

        # 5. Radar chart
        path = self._plot_radar_chart(summary, model_names, colors)
        if path:
            generated["radar_comparison.png"] = path

        # 6. Metrics heatmap
        path = self._plot_metrics_heatmap(summary, model_names)
        if path:
            generated["metrics_heatmap.png"] = path

        # 7. Rankings table visualization
        path = self._plot_rankings_table(summary, model_names, colors)
        if path:
            generated["rankings_table.png"] = path

        # 8. Deltas from baseline
        if summary.baseline_name:
            path = self._plot_deltas_chart(summary, model_names, colors)
            if path:
                generated["deltas_from_baseline.png"] = path

        return generated

    def _plot_summary_bars(
        self,
        summary: MultiModelComparisonSummary,
        model_names: list[str],
        colors: list[str],
    ) -> Path:
        """Generate 3-panel bar chart for key metrics."""
        n_panels = sum([
            bool(summary.perplexities),
            bool(summary.throughputs),
            bool(summary.kv_bytes_per_token),
        ])
        if n_panels == 0:
            n_panels = 1

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # Perplexity panel
        if summary.perplexities:
            ax = axes[panel_idx]
            values = [summary.perplexities.get(m, 0) for m in model_names]
            bars = ax.bar(model_names, values, color=colors, edgecolor="black", linewidth=1)
            ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
            ax.set_title("Language Modeling Quality", fontsize=14, fontweight="bold")
            ax.bar_label(bars, fmt="%.2f", fontsize=10)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            # Highlight baseline
            if summary.baseline_name and summary.baseline_name in model_names:
                idx = model_names.index(summary.baseline_name)
                bars[idx].set_edgecolor("gold")
                bars[idx].set_linewidth(3)
            panel_idx += 1

        # Throughput panel
        if summary.throughputs:
            ax = axes[panel_idx]
            values = [summary.throughputs.get(m, 0) for m in model_names]
            bars = ax.bar(model_names, values, color=colors, edgecolor="black", linewidth=1)
            ax.set_ylabel("Tokens/Second (↑ better)", fontsize=12)
            ax.set_title("Throughput", fontsize=14, fontweight="bold")
            ax.bar_label(bars, fmt="%.0f", fontsize=10)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            if summary.baseline_name and summary.baseline_name in model_names:
                idx = model_names.index(summary.baseline_name)
                bars[idx].set_edgecolor("gold")
                bars[idx].set_linewidth(3)
            panel_idx += 1

        # Memory panel
        if summary.kv_bytes_per_token:
            ax = axes[panel_idx]
            values = [summary.kv_bytes_per_token.get(m, 0) for m in model_names]
            bars = ax.bar(model_names, values, color=colors, edgecolor="black", linewidth=1)
            ax.set_ylabel("KV-Cache Bytes/Token (↓ better)", fontsize=12)
            ax.set_title("Memory Efficiency", fontsize=14, fontweight="bold")
            ax.bar_label(bars, fmt="%.0f", fontsize=10)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            if summary.baseline_name and summary.baseline_name in model_names:
                idx = model_names.index(summary.baseline_name)
                bars[idx].set_edgecolor("gold")
                bars[idx].set_linewidth(3)

        fig.set_constrained_layout(True)
        path = self.output_dir / "comparison_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_latency_vs_context(
        self,
        latency_results: dict[str, LatencyResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot throughput vs context length for all models."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(model_names):
            if name not in latency_results:
                continue
            result = latency_results[name]
            # Group by prompt_len and get average throughput
            by_prompt: dict[int, list[float]] = {}
            for m in result.measurements:
                if m.prompt_len not in by_prompt:
                    by_prompt[m.prompt_len] = []
                by_prompt[m.prompt_len].append(m.tokens_per_second)

            if not by_prompt:
                continue

            x = sorted(by_prompt.keys())
            y = [np.mean(by_prompt[pl]) for pl in x]
            ax.plot(x, y, "o-", color=colors[i], label=name, linewidth=2, markersize=8)

        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        ax.set_ylabel("Tokens/Second", fontsize=12)
        ax.set_title("Throughput vs Context Length", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.set_constrained_layout(True)
        path = self.output_dir / "latency_vs_context.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_memory_vs_context(
        self,
        memory_results: dict[str, MemoryResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot KV-cache memory vs context length for all models."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(model_names):
            if name not in memory_results:
                continue
            result = memory_results[name]
            # Get seq_len vs kvcache_memory
            by_seq: dict[int, list[float]] = {}
            for m in result.measurements:
                if m.seq_len not in by_seq:
                    by_seq[m.seq_len] = []
                by_seq[m.seq_len].append(m.kvcache_memory_mb)

            if not by_seq:
                continue

            x = sorted(by_seq.keys())
            y = [np.mean(by_seq[sl]) for sl in x]
            ax.plot(x, y, "o-", color=colors[i], label=name, linewidth=2, markersize=8)

        ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax.set_ylabel("KV-Cache Memory (MB)", fontsize=12)
        ax.set_title("Memory vs Context Length", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.set_constrained_layout(True)
        path = self.output_dir / "memory_vs_context.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_pareto_frontier(
        self,
        summary: MultiModelComparisonSummary,
        model_names: list[str],
        colors: list[str],
    ) -> Path:
        """Plot Pareto frontier of perplexity vs memory."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points
        for i, name in enumerate(model_names):
            if name not in summary.perplexities or name not in summary.kv_bytes_per_token:
                continue
            x = summary.kv_bytes_per_token[name]
            y = summary.perplexities[name]

            marker = "s" if name in summary.pareto_optimal else "o"
            size = 200 if name in summary.pareto_optimal else 120
            edgecolor = "gold" if name in summary.pareto_optimal else "black"
            linewidth = 3 if name in summary.pareto_optimal else 1

            ax.scatter(
                x, y, c=colors[i], s=size, marker=marker,
                edgecolors=edgecolor, linewidths=linewidth, label=name, zorder=10
            )
            ax.annotate(
                name, (x, y), xytext=(8, 8), textcoords="offset points",
                fontsize=11, fontweight="bold"
            )

        # Draw Pareto frontier line
        if len(summary.pareto_optimal) > 1:
            pareto_points = [
                (summary.kv_bytes_per_token[m], summary.perplexities[m])
                for m in summary.pareto_optimal
            ]
            pareto_points.sort()
            px, py = zip(*pareto_points)
            ax.plot(px, py, "g--", linewidth=2, alpha=0.7, label="Pareto Frontier", zorder=5)

        ax.set_xlabel("KV-Cache Bytes/Token (↓ better)", fontsize=12)
        ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
        ax.set_title("Quality vs Memory Tradeoff", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add annotations for Pareto optimal
        pareto_patch = mpatches.Patch(
            facecolor="none", edgecolor="gold", linewidth=2,
            label="Pareto Optimal (gold border)"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(pareto_patch)
        ax.legend(handles=handles, loc="upper right")

        fig.set_constrained_layout(True)
        path = self.output_dir / "pareto_frontier.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_radar_chart(
        self,
        summary: MultiModelComparisonSummary,
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Generate radar/spider chart for comprehensive comparison."""
        # Need at least 3 metrics for a meaningful radar chart
        metrics = []
        values_by_model: dict[str, list[float]] = {m: [] for m in model_names}

        # Normalize metrics to [0, 1] scale (best = 1.0)
        if summary.perplexities:
            metrics.append("Quality")
            best = min(summary.perplexities.values())
            for m in model_names:
                val = summary.perplexities.get(m)
                values_by_model[m].append(best / val if val else 0)

        if summary.throughputs:
            metrics.append("Speed")
            best = max(summary.throughputs.values())
            for m in model_names:
                val = summary.throughputs.get(m)
                values_by_model[m].append(val / best if best else 0)

        if summary.kv_bytes_per_token:
            metrics.append("Memory")
            best = min(summary.kv_bytes_per_token.values())
            for m in model_names:
                val = summary.kv_bytes_per_token.get(m)
                values_by_model[m].append(best / val if val else 0)

        if summary.efficiency_scores:
            metrics.append("Efficiency")
            best = max(summary.efficiency_scores.values())
            for m in model_names:
                val = summary.efficiency_scores.get(m)
                values_by_model[m].append(val / best if best else 0)

        if len(metrics) < 3:
            return None

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for i, m in enumerate(model_names):
            vals = values_by_model[m] + values_by_model[m][:1]  # Complete circle
            ax.plot(angles, vals, "o-", linewidth=2, label=m, color=colors[i], markersize=6)
            ax.fill(angles, vals, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.set_title("Multi-Dimensional Comparison\n(normalized, 100% = best)", fontsize=14, fontweight="bold", pad=20)

        fig.set_constrained_layout(True)
        path = self.output_dir / "radar_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_metrics_heatmap(
        self,
        summary: MultiModelComparisonSummary,
        model_names: list[str],
    ) -> Path | None:
        """Generate heatmap of all metrics × models."""
        # Build data matrix
        metrics = []
        data = []

        if summary.perplexities:
            metrics.append("Perplexity")
            best = min(summary.perplexities.values())
            row = [best / summary.perplexities.get(m, best) if summary.perplexities.get(m) else 0 for m in model_names]
            data.append(row)

        if summary.throughputs:
            metrics.append("Throughput")
            best = max(summary.throughputs.values())
            row = [summary.throughputs.get(m, 0) / best if best else 0 for m in model_names]
            data.append(row)

        if summary.kv_bytes_per_token:
            metrics.append("Memory Eff.")
            best = min(summary.kv_bytes_per_token.values())
            row = [best / summary.kv_bytes_per_token.get(m, best) if summary.kv_bytes_per_token.get(m) else 0 for m in model_names]
            data.append(row)

        if summary.efficiency_scores:
            metrics.append("Overall Eff.")
            best = max(summary.efficiency_scores.values())
            row = [summary.efficiency_scores.get(m, 0) / best if best else 0 for m in model_names]
            data.append(row)

        if not data:
            return None

        data = np.array(data)

        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5), len(metrics) * 1.2 + 1))

        # Custom colormap: red-yellow-green
        cmap = LinearSegmentedColormap.from_list("ryg", ["#e74c3c", "#f39c12", "#2ecc71"])
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels(model_names, fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(metrics, fontsize=11)

        # Add value annotations
        for i in range(len(metrics)):
            for j in range(len(model_names)):
                val = data[i, j]
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Normalized Score (100% = best)", fontsize=11)

        ax.set_title("Metrics Comparison Heatmap", fontsize=14, fontweight="bold", pad=10)

        fig.set_constrained_layout(True)
        path = self.output_dir / "metrics_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_rankings_table(
        self,
        summary: MultiModelComparisonSummary,
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Generate visual rankings table."""
        rankings_data = []
        headers = ["Metric"] + model_names

        if summary.perplexity_rankings:
            row = ["Perplexity"]
            for m in model_names:
                rank = summary.get_rank(m, "perplexity")
                row.append(f"#{rank}" if rank else "-")
            rankings_data.append(row)

        if summary.throughput_rankings:
            row = ["Throughput"]
            for m in model_names:
                rank = summary.get_rank(m, "throughput")
                row.append(f"#{rank}" if rank else "-")
            rankings_data.append(row)

        if summary.memory_rankings:
            row = ["Memory"]
            for m in model_names:
                rank = summary.get_rank(m, "memory")
                row.append(f"#{rank}" if rank else "-")
            rankings_data.append(row)

        if not rankings_data:
            return None

        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), len(rankings_data) * 0.8 + 1.5))
        ax.axis("off")

        table = ax.table(
            cellText=rankings_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colColours=["#ecf0f1"] + colors[:len(model_names)],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        # Color cells by rank
        for (row, col), cell in table.get_celld().items():
            if row > 0 and col > 0:  # Data cells
                text = cell.get_text().get_text()
                if text == "#1":
                    cell.set_facecolor("#2ecc71")  # Green for first
                    cell.get_text().set_color("white")
                elif text == "#2":
                    cell.set_facecolor("#f39c12")  # Yellow for second
                elif text == "#3":
                    cell.set_facecolor("#e74c3c")  # Red for third
                    cell.get_text().set_color("white")

        ax.set_title("Rankings by Metric\n(#1 = best)", fontsize=14, fontweight="bold", pad=20)

        fig.set_constrained_layout(True)
        path = self.output_dir / "rankings_table.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _plot_deltas_chart(
        self,
        summary: MultiModelComparisonSummary,
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot deltas from baseline for all metrics."""
        if not summary.baseline_name:
            return None

        # Collect delta data
        metrics = []
        deltas_by_metric: dict[str, dict[str, float]] = {}

        if summary.perplexity_deltas:
            metrics.append("PPL Delta")
            deltas_by_metric["PPL Delta"] = summary.perplexity_deltas

        if summary.speedups:
            metrics.append("Speedup")
            # Convert to percentage change
            deltas_by_metric["Speedup"] = {
                m: (v - 1.0) * 100 for m, v in summary.speedups.items()
            }

        if summary.memory_reductions:
            metrics.append("Memory Red.")
            # Convert to percentage change
            deltas_by_metric["Memory Red."] = {
                m: (v - 1.0) * 100 for m, v in summary.memory_reductions.items()
            }

        if not metrics:
            return None

        # Remove baseline from comparison
        compare_models = [m for m in model_names if m != summary.baseline_name]
        if not compare_models:
            return None

        fig, ax = plt.subplots(figsize=(max(8, len(compare_models) * 2), 6))

        x = np.arange(len(compare_models))
        width = 0.8 / len(metrics)

        for i, metric in enumerate(metrics):
            vals = [deltas_by_metric[metric].get(m, 0) for m in compare_models]
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=metric, alpha=0.8)

            # Color bars based on positive/negative (green = better)
            for bar, val in zip(bars, vals):
                if metric == "PPL Delta":
                    # Lower is better, so negative delta is good
                    bar.set_color("#2ecc71" if val < 0 else "#e74c3c")
                else:
                    # Higher is better, so positive is good
                    bar.set_color("#2ecc71" if val > 0 else "#e74c3c")

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Change from Baseline", fontsize=12)
        ax.set_title(f"Improvements vs Baseline ({summary.baseline_name})", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(compare_models, fontsize=11)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        fig.set_constrained_layout(True)
        path = self.output_dir / "deltas_from_baseline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def _write_latex_tables(
        self,
        metadata: ExperimentMetadata,
        summary: MultiModelComparisonSummary,
        perplexity_results: dict[str, PerplexityResult] | None = None,
        latency_results: dict[str, LatencyResult] | None = None,
        memory_results: dict[str, MemoryResult] | None = None,
        accuracy_results: dict[str, AccuracyResult] | None = None,
        context_results: dict[str, ContextResult] | None = None,
    ) -> dict[str, Path]:
        """Write comprehensive LaTeX tables for paper inclusion."""
        generated: dict[str, Path] = {}

        # ========================================================================
        # 1. MAIN COMPARISON TABLE (tables.tex)
        # ========================================================================
        lines = [
            "% Auto-generated LaTeX tables for multi-model comparison",
            f"% Generated: {datetime.now().isoformat()}",
            f"% Experiment: {metadata.name}",
            "",
            "% ============================================================================",
            "% MAIN COMPARISON TABLE",
            "% ============================================================================",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Multi-model benchmark comparison}",
            "\\label{tab:multi-model-comparison}",
            f"\\begin{{tabular}}{{l{'r' * len(summary.model_names)}}}",
            "\\toprule",
        ]

        # Header row
        header = "Metric & " + " & ".join(summary.model_names) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Perplexity row
        if summary.perplexities:
            vals = [f"{summary.perplexities.get(m, 0):.2f}" for m in summary.model_names]
            best_idx = summary.perplexity_rankings.index(summary.perplexity_rankings[0]) if summary.perplexity_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"Perplexity ($\\downarrow$) & {' & '.join(vals)} \\\\")

        # Throughput row
        if summary.throughputs:
            vals = [f"{summary.throughputs.get(m, 0):.0f}" for m in summary.model_names]
            best_idx = summary.throughput_rankings.index(summary.throughput_rankings[0]) if summary.throughput_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"Tokens/sec ($\\uparrow$) & {' & '.join(vals)} \\\\")

        # Memory row
        if summary.kv_bytes_per_token:
            vals = [f"{summary.kv_bytes_per_token.get(m, 0):.0f}" for m in summary.model_names]
            best_idx = summary.memory_rankings.index(summary.memory_rankings[0]) if summary.memory_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"KV Bytes/tok ($\\downarrow$) & {' & '.join(vals)} \\\\")

        # Efficiency row
        if summary.efficiency_scores:
            vals = [f"{summary.efficiency_scores.get(m, 0):.2f}" for m in summary.model_names]
            lines.append(f"Efficiency Score & {' & '.join(vals)} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

        # Add deltas table if baseline exists
        if summary.baseline_name and summary.perplexity_deltas:
            lines.extend([
                "% ============================================================================",
                "% DELTAS FROM BASELINE TABLE",
                "% ============================================================================",
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{Improvements vs baseline ({summary.baseline_name})}}",
                "\\label{tab:deltas}",
            ])
            compare_models = [m for m in summary.model_names if m != summary.baseline_name]
            lines.append(f"\\begin{{tabular}}{{l{'r' * len(compare_models)}}}")
            lines.append("\\toprule")
            lines.append("Metric & " + " & ".join(compare_models) + " \\\\")
            lines.append("\\midrule")

            if summary.perplexity_deltas:
                vals = [f"{summary.perplexity_deltas.get(m, 0):+.2f}" for m in compare_models]
                lines.append(f"PPL $\\Delta$ & {' & '.join(vals)} \\\\")

            if summary.speedups:
                vals = [f"{(summary.speedups.get(m, 1) - 1) * 100:+.1f}\\%" for m in compare_models]
                lines.append(f"Speedup & {' & '.join(vals)} \\\\")

            if summary.memory_reductions:
                vals = [f"{summary.memory_reductions.get(m, 1):.2f}$\\times$" for m in compare_models]
                lines.append(f"Mem. reduction & {' & '.join(vals)} \\\\")

            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])

        path = self.output_dir / "tables.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        generated["tables.tex"] = path

        # ========================================================================
        # 2. DETAILED PERPLEXITY TABLE (table_perplexity.tex)
        # ========================================================================
        if perplexity_results:
            ppl_lines = [
                "% Detailed perplexity results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Perplexity Results (FineWeb validation)}",
                "\\label{tab:perplexity-detailed}",
                "\\begin{tabular}{lrrrr}",
                "\\toprule",
                "Model & Perplexity & Loss & Tokens & Batches \\\\",
                "\\midrule",
            ]
            for name in summary.model_names:
                if name in perplexity_results:
                    r = perplexity_results[name]
                    ppl_lines.append(f"{name} & {r.perplexity:.2f} & {r.loss:.4f} & {r.num_tokens:,} & {r.num_batches} \\\\")
            ppl_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_perplexity.tex"
            with open(path, "w") as f:
                f.write("\n".join(ppl_lines))
            generated["table_perplexity.tex"] = path

        # ========================================================================
        # 3. LATENCY/THROUGHPUT TABLE (table_latency.tex)
        # ========================================================================
        if latency_results:
            lat_lines = [
                "% Latency and throughput results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Throughput by Context Length}",
                "\\label{tab:latency-detailed}",
                "\\begin{tabular}{l" + "r" * len(summary.model_names) + "}",
                "\\toprule",
                "Context & " + " & ".join(summary.model_names) + " \\\\",
                " & " + " & ".join(["(tok/s)"] * len(summary.model_names)) + " \\\\",
                "\\midrule",
            ]

            # Collect all unique prompt lengths
            all_prompt_lens: set[int] = set()
            for r in latency_results.values():
                for m in r.measurements:
                    all_prompt_lens.add(m.prompt_len)

            for prompt_len in sorted(all_prompt_lens):
                row_vals = []
                for name in summary.model_names:
                    if name in latency_results:
                        r = latency_results[name]
                        tps_vals = [m.tokens_per_second for m in r.measurements if m.prompt_len == prompt_len]
                        if tps_vals:
                            avg_tps = np.mean(tps_vals)
                            row_vals.append(f"{avg_tps:.0f}")
                        else:
                            row_vals.append("--")
                    else:
                        row_vals.append("--")
                lat_lines.append(f"{prompt_len} & {' & '.join(row_vals)} \\\\")

            lat_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_latency.tex"
            with open(path, "w") as f:
                f.write("\n".join(lat_lines))
            generated["table_latency.tex"] = path

            # Also create a summary latency table
            lat_summary_lines = [
                "% Latency summary",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Latency Summary}",
                "\\label{tab:latency-summary}",
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "Model & Avg Tok/s & TTFT (ms) & Prefill (ms) \\\\",
                "\\midrule",
            ]
            for name in summary.model_names:
                if name in latency_results:
                    r = latency_results[name]
                    avg_tps = r.avg_tokens_per_second
                    avg_ttft = r.avg_time_to_first_token_ms
                    avg_prefill = np.mean([m.prefill_time_ms for m in r.measurements]) if r.measurements else 0
                    lat_summary_lines.append(f"{name} & {avg_tps:.0f} & {avg_ttft:.1f} & {avg_prefill:.1f} \\\\")
            lat_summary_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_latency_summary.tex"
            with open(path, "w") as f:
                f.write("\n".join(lat_summary_lines))
            generated["table_latency_summary.tex"] = path

        # ========================================================================
        # 4. MEMORY TABLE (table_memory.tex)
        # ========================================================================
        if memory_results:
            mem_lines = [
                "% Memory efficiency results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{KV-Cache Memory by Sequence Length}",
                "\\label{tab:memory-detailed}",
                "\\begin{tabular}{l" + "r" * len(summary.model_names) + "}",
                "\\toprule",
                "Seq Len & " + " & ".join(summary.model_names) + " \\\\",
                " & " + " & ".join(["(MB)"] * len(summary.model_names)) + " \\\\",
                "\\midrule",
            ]

            # Collect all unique sequence lengths
            all_seq_lens: set[int] = set()
            for r in memory_results.values():
                for m in r.measurements:
                    all_seq_lens.add(m.seq_len)

            for seq_len in sorted(all_seq_lens):
                row_vals = []
                for name in summary.model_names:
                    if name in memory_results:
                        r = memory_results[name]
                        mem_vals = [m.kvcache_memory_mb for m in r.measurements if m.seq_len == seq_len]
                        if mem_vals:
                            avg_mem = np.mean(mem_vals)
                            row_vals.append(f"{avg_mem:.1f}")
                        else:
                            row_vals.append("--")
                    else:
                        row_vals.append("--")
                mem_lines.append(f"{seq_len} & {' & '.join(row_vals)} \\\\")

            mem_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_memory.tex"
            with open(path, "w") as f:
                f.write("\n".join(mem_lines))
            generated["table_memory.tex"] = path

            # KV bytes per token table
            kv_lines = [
                "% KV-cache bytes per token",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{KV-Cache Bytes per Token}",
                "\\label{tab:kv-bytes}",
                "\\begin{tabular}{lrr}",
                "\\toprule",
                "Model & Bytes/Token & Reduction \\\\",
                "\\midrule",
            ]
            baseline_bytes = summary.kv_bytes_per_token.get(summary.baseline_name, 0) if summary.baseline_name else 0
            for name in summary.model_names:
                bytes_per_tok = summary.kv_bytes_per_token.get(name, 0)
                if baseline_bytes > 0 and name != summary.baseline_name:
                    reduction = baseline_bytes / bytes_per_tok if bytes_per_tok > 0 else 0
                    kv_lines.append(f"{name} & {bytes_per_tok:.0f} & {reduction:.2f}$\\times$ \\\\")
                else:
                    kv_lines.append(f"{name} & {bytes_per_tok:.0f} & -- \\\\")
            kv_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_kv_bytes.tex"
            with open(path, "w") as f:
                f.write("\n".join(kv_lines))
            generated["table_kv_bytes.tex"] = path

        # ========================================================================
        # 5. ACCURACY TABLE (table_accuracy.tex)
        # ========================================================================
        if accuracy_results:
            acc_lines = [
                "% Downstream task accuracy",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Downstream Task Accuracy}",
                "\\label{tab:accuracy}",
            ]

            # Collect all tasks
            all_tasks: set[str] = set()
            for r in accuracy_results.values():
                for t in r.tasks:
                    all_tasks.add(t.task)
            all_tasks_sorted = sorted(all_tasks)

            acc_lines.append(f"\\begin{{tabular}}{{l{'r' * len(summary.model_names)}}}")
            acc_lines.append("\\toprule")
            acc_lines.append("Task & " + " & ".join(summary.model_names) + " \\\\")
            acc_lines.append("\\midrule")

            for task in all_tasks_sorted:
                row_vals = []
                task_accs = []
                for name in summary.model_names:
                    if name in accuracy_results:
                        r = accuracy_results[name]
                        task_result = next((t for t in r.tasks if t.task == task), None)
                        if task_result:
                            task_accs.append((name, task_result.accuracy))
                            row_vals.append(f"{task_result.accuracy * 100:.1f}\\%")
                        else:
                            row_vals.append("--")
                    else:
                        row_vals.append("--")

                # Bold the best
                if task_accs:
                    best_name = max(task_accs, key=lambda x: x[1])[0]
                    for i, name in enumerate(summary.model_names):
                        if name == best_name:
                            row_vals[i] = f"\\textbf{{{row_vals[i]}}}"

                acc_lines.append(f"{task} & {' & '.join(row_vals)} \\\\")

            # Add micro accuracy row
            acc_lines.append("\\midrule")
            micro_vals = []
            micro_accs = []
            for name in summary.model_names:
                if name in accuracy_results:
                    r = accuracy_results[name]
                    micro_accs.append((name, r.micro_accuracy))
                    micro_vals.append(f"{r.micro_accuracy * 100:.1f}\\%")
                else:
                    micro_vals.append("--")
            if micro_accs:
                best_name = max(micro_accs, key=lambda x: x[1])[0]
                for i, name in enumerate(summary.model_names):
                    if name == best_name:
                        micro_vals[i] = f"\\textbf{{{micro_vals[i]}}}"
            acc_lines.append(f"\\textit{{Average}} & {' & '.join(micro_vals)} \\\\")

            acc_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_accuracy.tex"
            with open(path, "w") as f:
                f.write("\n".join(acc_lines))
            generated["table_accuracy.tex"] = path

        # ========================================================================
        # 6. RANKINGS TABLE (table_rankings.tex)
        # ========================================================================
        rank_lines = [
            "% Model rankings by metric",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Rankings by Metric}",
            "\\label{tab:rankings}",
            f"\\begin{{tabular}}{{l{'c' * len(summary.model_names)}}}",
            "\\toprule",
            "Metric & " + " & ".join(summary.model_names) + " \\\\",
            "\\midrule",
        ]

        if summary.perplexity_rankings:
            row_vals = []
            for name in summary.model_names:
                rank = summary.get_rank(name, "perplexity")
                row_vals.append(f"\\#{rank}" if rank else "--")
            rank_lines.append(f"Perplexity & {' & '.join(row_vals)} \\\\")

        if summary.throughput_rankings:
            row_vals = []
            for name in summary.model_names:
                rank = summary.get_rank(name, "throughput")
                row_vals.append(f"\\#{rank}" if rank else "--")
            rank_lines.append(f"Throughput & {' & '.join(row_vals)} \\\\")

        if summary.memory_rankings:
            row_vals = []
            for name in summary.model_names:
                rank = summary.get_rank(name, "memory")
                row_vals.append(f"\\#{rank}" if rank else "--")
            rank_lines.append(f"Memory Eff. & {' & '.join(row_vals)} \\\\")

        if summary.accuracy_rankings:
            row_vals = []
            for name in summary.model_names:
                rank = summary.get_rank(name, "accuracy")
                row_vals.append(f"\\#{rank}" if rank else "--")
            rank_lines.append(f"Accuracy & {' & '.join(row_vals)} \\\\")

        rank_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        path = self.output_dir / "table_rankings.tex"
        with open(path, "w") as f:
            f.write("\n".join(rank_lines))
        generated["table_rankings.tex"] = path

        # ========================================================================
        # 7. CONTEXT SWEEP TABLE (table_context_sweep.tex)
        # ========================================================================
        if context_results:
            ctx_lines = [
                "% Context sweep results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Decode Throughput by Context Length}",
                "\\label{tab:context-sweep}",
                f"\\begin{{tabular}}{{r{'r' * len(summary.model_names)}}}",
                "\\toprule",
                "Context & " + " & ".join(summary.model_names) + " \\\\",
                "(tokens) & " + " & ".join(["(tok/s)"] * len(summary.model_names)) + " \\\\",
                "\\midrule",
            ]

            # Collect all context lengths
            all_ctx_lens: set[int] = set()
            for r in context_results.values():
                for m in r.decode:
                    all_ctx_lens.add(m.seq_len)

            for ctx_len in sorted(all_ctx_lens):
                row_vals = []
                for name in summary.model_names:
                    if name in context_results:
                        r = context_results[name]
                        tps_vals = [m.decode_tok_per_s for m in r.decode if m.seq_len == ctx_len and m.ok]
                        if tps_vals:
                            avg_tps = np.mean(tps_vals)
                            row_vals.append(f"{avg_tps:.0f}")
                        else:
                            row_vals.append("--")
                    else:
                        row_vals.append("--")
                ctx_lines.append(f"{ctx_len:,} & {' & '.join(row_vals)} \\\\")

            ctx_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_context_sweep.tex"
            with open(path, "w") as f:
                f.write("\n".join(ctx_lines))
            generated["table_context_sweep.tex"] = path

        return generated
