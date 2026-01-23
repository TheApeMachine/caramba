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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from benchmark.accuracy import AccuracyResult
from benchmark.artifacts import ExperimentMetadata
from benchmark.context import ContextResult
from benchmark.latency import LatencyResult
from benchmark.memory import MemoryResult
from benchmark.multi_model_results import (
    MultiModelComparisonSummary,
    build_comparison_summary,
)
from benchmark.perplexity import PerplexityResult
from benchmark.perplexity_microscope import write_microscope
from console import logger


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

def _jsonable(x: Any) -> Any:
    """Convert a nested structure into JSON-serializable primitives.

    This is intentionally strict: we convert known numeric wrapper types
    (e.g. NumPy scalars) to Python scalars, and otherwise raise.
    """
    # Primitives
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    # NumPy scalar / arrays
    try:
        import numpy as _np  # type: ignore

        if isinstance(x, _np.generic):
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    # Mappings / sequences
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    # Dataclasses
    try:
        from dataclasses import is_dataclass, asdict as _asdict

        # dataclasses.is_dataclass returns true for instances and classes; asdict needs instances.
        if is_dataclass(x) and not isinstance(x, type):
            return _jsonable(_asdict(x))
    except Exception:
        pass
    # Paths
    try:
        from pathlib import Path

        if isinstance(x, Path):
            return str(x)
    except Exception:
        pass
    raise TypeError(f"Object of type {type(x).__name__} is not JSON-serializable")


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
        audit: dict | None = None,
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
                memory_results, accuracy_results, context_results, behavioral_results,
                audit=audit,
            )
            generated.update(paths)

        if "csv" in formats:
            paths = self._write_csv_files(
                summary, perplexity_results, latency_results,
                memory_results, accuracy_results, context_results
            )
            generated.update(paths)


        if "png" in formats:
            paths = self._generate_charts(
                summary, perplexity_results, latency_results,
                memory_results, accuracy_results, context_results, behavioral_results
            )
            generated.update(paths)

        if "latex" in formats:
            paths = self._write_latex_tables(
                metadata, summary,
                perplexity_results=perplexity_results,
                latency_results=latency_results,
                memory_results=memory_results,
                accuracy_results=accuracy_results,
                context_results=context_results,
                behavioral_results=behavioral_results,
            )
            generated.update(paths)

        # Perplexity microscope sidecar (per-model batch CSV + summary JSON).
        # Best-effort: never fail the full artifact generation.
        try:
            if perplexity_results:
                for name, r in perplexity_results.items():
                    if getattr(r, "batch_loss_sums", None):
                        generated.update(
                            write_microscope(output_dir=self.output_dir, result=r, prefix=str(name))
                        )
        except Exception:
            pass

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
        *,
        audit: dict | None = None,
    ) -> dict[str, Path]:
        """Write comprehensive JSON report."""
        report: dict[str, Any] = {
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
                    "batch_loss_sums": list(getattr(r, "batch_loss_sums", [])),
                    "batch_token_counts": list(getattr(r, "batch_token_counts", [])),
                    "measurements": [
                        {
                            "perplexity": m.perplexity,
                            "loss": m.loss,
                            "num_tokens": m.num_tokens,
                        }
                        for m in getattr(r, "measurements", [])
                    ]
                }
                for name, r in perplexity_results.items()
            }

        if latency_results:
            report["detailed_results"]["latency"] = {
                name: {
                    "avg_tokens_per_second": r.avg_tokens_per_second,
                    "avg_time_to_first_token_ms": r.avg_time_to_first_token_ms,
                    # Full audit trail per measurement (includes seed, inputs, and raw timed runs).
                    "measurements": [asdict(m) for m in r.measurements],
                }
                for name, r in latency_results.items()
            }

        if memory_results:
            report["detailed_results"]["memory"] = {
                name: {
                    "peak_memory_mb": r.peak_memory_mb,
                    "kvcache_analysis": (asdict(r.kvcache_analysis) if r.kvcache_analysis else None),
                    # Full audit trail per measurement (includes seed, inputs, and backend telemetry).
                    "measurements": [asdict(m) for m in r.measurements],
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

        if context_results:
            report["detailed_results"]["context"] = {
                name: {
                    "sweep": [asdict(m) for m in r.sweep],
                    "decode": [asdict(m) for m in r.decode],
                }
                for name, r in context_results.items()
            }

        if audit is not None:
            report["audit"] = audit

        path = self.output_dir / "report.json"
        with open(path, "w") as f:
            json.dump(_jsonable(report), f, indent=2)

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

        # Context sweep CSV [NEW]
        if context_results:
            path = self.output_dir / "context_sweep.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "context_len", "decode_tps", "is_ok"])
                for name, r in context_results.items():
                    for m in r.decode:
                        writer.writerow([name, m.context_len, m.decode_tok_per_s, m.ok])
            generated["context_sweep.csv"] = path

        return generated

    def _write_accuracy_log(
        self,
        accuracy_results: dict[str, AccuracyResult],
    ) -> Path | None:
        """Write detailed accuracy log with raw samples."""
        log_lines: list[str] = []
        log_lines.append("=" * 80)
        log_lines.append(f"ACCURACY BENCHMARK LOG")
        log_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append("=" * 80)
        log_lines.append("")

        # Sort tasks to be consistent
        all_tasks = set()
        for r in accuracy_results.values():
            for t in r.tasks:
                all_tasks.add(t.task)
        sorted_tasks = sorted(all_tasks)

        for task_name in sorted_tasks:
            for model_name, res in accuracy_results.items():
                t_res = next((t for t in res.tasks if t.task == task_name), None)
                if not t_res:
                    continue

                log_lines.append("=" * 80)
                log_lines.append(f"TASK: {task_name} | MODEL: {model_name}")
                log_lines.append("=" * 80)
                log_lines.append("")

                if not t_res.samples:
                    log_lines.append("(No samples collected)")
                    log_lines.append("")
                    continue

                for i, s in enumerate(t_res.samples):
                    mark = "✓" if s.ok else "✗"
                    log_lines.append(f"[{i+1}] {mark}")
                    log_lines.append("-" * 40)
                    log_lines.append("PROMPT:")
                    log_lines.append(s.prompt.strip())
                    log_lines.append("")
                    log_lines.append(f"GOLD: {s.gold.strip()}")
                    log_lines.append(f"PRED: {s.pred.strip()}")
                    log_lines.append("")

                log_lines.append("")

        path = self.output_dir / "accuracy_log.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        return path

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

        # 9. Accuracy summary [NEW]
        if accuracy_results:
            path = self._plot_accuracy_summary(accuracy_results, model_names, colors)
            if path:
                generated["accuracy_summary.png"] = path

        # 10. Latency breakdown (TTFT vs Decode) [NEW]
        if latency_results:
            path = self._plot_latency_breakdown(latency_results, model_names, colors)
            if path:
                generated["latency_breakdown.png"] = path

        # 11. Perplexity comparison (PPL vs Loss) [NEW]
        if summary.perplexities:
            path = self._plot_perplexity_comparison(summary, perplexity_results, model_names, colors)
            if path:
                generated["perplexity_comparison.png"] = path

        # 12. Behavioral summary [NEW]
        if behavioral_results:
            path = self._plot_behavioral_summary(behavioral_results, model_names, colors)
            if path:
                generated["behavioral_summary.png"] = path

            path = self._plot_behavioral_categories(behavioral_results, model_names, colors)
            if path:
                generated["behavioral_categories.png"] = path

        # 13. Accuracy per task [NEW]
        if accuracy_results:
            path = self._plot_accuracy_per_task(accuracy_results, model_names, colors)
            if path:
                generated["accuracy_per_task.png"] = path

            path = self._plot_accuracy_radar(accuracy_results, model_names, colors)
            if path:
                generated["accuracy_radar.png"] = path

            path = self._plot_accuracy_heat_map(accuracy_results, model_names)
            if path:
                generated["accuracy_heatmap.png"] = path

            # 14. Accuracy timing comparison
            path = self._plot_accuracy_timing(accuracy_results, model_names, colors)
            if path:
                generated["accuracy_timing.png"] = path

        # 15. Context sweep chart [NEW]
        if context_results:
            path = self._plot_context_sweep_chart(context_results, model_names, colors)
            if path:
                generated["context_sweep.png"] = path

        # 16. Overall comparison radar (aggregated across all benchmarks)
        path = self._plot_overall_comparison(
            perplexity_results=perplexity_results,
            memory_results=memory_results,
            accuracy_results=accuracy_results,
            latency_results=latency_results,
            behavioral_results=behavioral_results,
            model_names=model_names,
            colors=colors,
        )
        if path:
            generated["overall_comparison.png"] = path

        # 17. Efficiency frontier (accuracy vs speed/memory trade-off)
        path = self._plot_efficiency_frontier(
            accuracy_results=accuracy_results,
            latency_results=latency_results,
            memory_results=memory_results,
            model_names=model_names,
            colors=colors,
        )
        if path:
            generated["efficiency_frontier.png"] = path

        # 18. Relative improvement vs baseline
        path = self._plot_relative_improvement(
            perplexity_results=perplexity_results,
            memory_results=memory_results,
            accuracy_results=accuracy_results,
            latency_results=latency_results,
            model_names=model_names,
            colors=colors,
            baseline_name=model_names[0] if model_names else None,
        )
        if path:
            generated["relative_improvement.png"] = path

        return generated

    def _plot_accuracy_summary(
        self,
        accuracy_results: dict[str, AccuracyResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot accuracy summary across tasks."""
        # Collect all tasks
        all_tasks: set[str] = set()
        for r in accuracy_results.values():
            for t in r.tasks:
                all_tasks.add(t.task)

        tasks_sorted = sorted(all_tasks)
        if not tasks_sorted:
            return None

        fig, ax = plt.subplots(figsize=(max(8.0, len(tasks_sorted) * 2), 6))

        x = np.arange(len(tasks_sorted))
        width = 0.8 / len(model_names)

        for i, name in enumerate(model_names):
            if name in accuracy_results:
                r = accuracy_results[name]
                vals = []
                for t_name in tasks_sorted:
                    t_res = next((t for t in r.tasks if t.task == t_name), None)
                    vals.append(t_res.accuracy * 100 if t_res else 0)

                offset = (i - len(model_names) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=name, color=colors[i], alpha=0.8)

        ax.set_ylabel("Accuracy (%)", fontsize=12)
        # No title (paper-friendly).
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_sorted, rotation=30, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        path = self.output_dir / "accuracy_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_latency_breakdown(
        self,
        latency_results: dict[str, LatencyResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot scatter of TTFT vs Decode Speed."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, name in enumerate(model_names):
            if name in latency_results:
                r = latency_results[name]
                ttft = r.avg_time_to_first_token_ms
                tps = r.avg_tokens_per_second

                ax.scatter(ttft, tps, s=200, label=name, color=colors[i], alpha=0.7, edgecolors="white")
                ax.annotate(name, (ttft, tps), xytext=(5, 5), textcoords="offset points", fontsize=10)

        ax.set_xlabel("Time to First Token (ms) - Lower is Better", fontsize=12)
        ax.set_ylabel("Decode Speed (tok/s) - Higher is Better", fontsize=12)
        # No title (paper-friendly).
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        path = self.output_dir / "latency_breakdown.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_perplexity_comparison(
        self,
        summary: MultiModelComparisonSummary,
        perplexity_results: dict[str, PerplexityResult] | None,
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot PPL and Loss comparison."""
        if not summary.perplexities or not perplexity_results:
            return None

        fig, ax1 = plt.subplots(figsize=(8, 6))

        x = np.arange(len(model_names))
        width = 0.35

        ppl_vals = [summary.perplexities.get(m, 0) for m in model_names]
        loss_vals = [perplexity_results[m].loss if m in perplexity_results else 0 for m in model_names]

        bar1 = ax1.bar(x - width/2, ppl_vals, width, label="Perplexity", color="#3498db", alpha=0.7)
        ax1.set_ylabel("Perplexity", color="#3498db", fontsize=12)
        ax1.tick_params(axis='y', labelcolor="#3498db")

        ax2 = ax1.twinx()
        bar2 = ax2.bar(x + width/2, loss_vals, width, label="Loss", color="#e67e22", alpha=0.7)
        ax2.set_ylabel("Cross Entropy Loss", color="#e67e22", fontsize=12)
        ax2.tick_params(axis='y', labelcolor="#e67e22")

        # No title (paper-friendly).
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=30, ha="right")

        # Combined legend
        lines = [bar1, bar2]
        labels = [l.get_label() for l in lines]
        ax1.legend(handles=lines, labels=labels, loc='upper right')

        fig.tight_layout()
        path = self.output_dir / "perplexity_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_behavioral_summary(
        self,
        behavioral_results: dict[str, dict],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot overall behavioral match counts."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data: Exact, Partial, None
        exacts = []
        partials = []
        nones = []

        for name in model_names:
            if name in behavioral_results:
                sum_data = behavioral_results[name].get("summary", {})
                # Handle multiple naming conventions across different scorers
                exacts.append(sum_data.get("exact_count", sum_data.get("exact_match_count", 0)))
                partials.append(sum_data.get("contained_count", sum_data.get("partial_count", sum_data.get("partial_match_count", 0))))
                nones.append(sum_data.get("none_count", sum_data.get("no_match_count", 0)))
            else:
                exacts.append(0)
                partials.append(0)
                nones.append(0)

        x = np.arange(len(model_names))

        ax.bar(x, exacts, label="Exact Match", color="#2ecc71", alpha=0.8)
        ax.bar(x, partials, bottom=exacts, label="Contained", color="#f39c12", alpha=0.8)
        ax.bar(x, nones, bottom=np.array(exacts) + np.array(partials), label="No Match", color="#e74c3c", alpha=0.8)

        ax.set_ylabel("Number of Test Cases", fontsize=12)
        # No title (paper-friendly).
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        path = self.output_dir / "behavioral_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_behavioral_categories(
        self,
        behavioral_results: dict[str, dict],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot behavioral scores by category."""
        # Collect all categories
        all_cats: set[str] = set()
        for r in behavioral_results.values():
            if "by_category" in r:
                all_cats.update(r["by_category"].keys())

        cats_sorted = sorted(all_cats)
        if not cats_sorted:
            return None

        # Build heatmap data
        data = np.zeros((len(model_names), len(cats_sorted)))
        for i, m_name in enumerate(model_names):
            if m_name in behavioral_results:
                cat_data = behavioral_results[m_name].get("by_category", {})
                for j, cat in enumerate(cats_sorted):
                    if cat in cat_data:
                        # Check if it's a dict or just a float
                        val = cat_data[cat]
                        if isinstance(val, dict):
                            # Handle multiple naming conventions for accuracy
                            # Try: soft_accuracy, partial_or_better_rate, exact_rate
                            acc = val.get("soft_accuracy",
                                   val.get("partial_or_better_rate",
                                   val.get("exact_rate", 0)))
                            data[i, j] = acc * 100
                        else:
                            data[i, j] = float(val) * 100

        fig, ax = plt.subplots(figsize=(len(cats_sorted) * 1.5 + 2, len(model_names) * 0.8 + 2))
        im = ax.imshow(data, cmap="YlGn")

        # Labels
        ax.set_xticks(np.arange(len(cats_sorted)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(cats_sorted, rotation=45, ha="right")
        ax.set_yticklabels(model_names)

        # Annotate cells
        for i in range(len(model_names)):
            for j in range(len(cats_sorted)):
                ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center", color="black" if data[i, j] < 70 else "white")

        # No title (paper-friendly).
        plt.colorbar(im, label="Soft Accuracy (%)", ax=ax, shrink=0.8)

        fig.tight_layout()
        path = self.output_dir / "behavioral_categories.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

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
        x = np.arange(len(model_names))

        # Perplexity panel
        if summary.perplexities:
            ax = axes[panel_idx]
            values = [summary.perplexities.get(m, 0) for m in model_names]
            bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1)
            ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
            # No title (paper-friendly).
            ax.bar_label(bars, fmt="%.2f", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            # Highlight best (lowest perplexity is better)
            valid = [(i, v) for i, v in enumerate(values) if v and v > 0]
            if valid:
                best_idx = min(valid, key=lambda iv: iv[1])[0]
                bars[best_idx].set_edgecolor("gold")
                bars[best_idx].set_linewidth(3)
            panel_idx += 1

        # Throughput panel
        if summary.throughputs:
            ax = axes[panel_idx]
            values = [summary.throughputs.get(m, 0) for m in model_names]
            bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1)
            ax.set_ylabel("Tokens/Second (↑ better)", fontsize=12)
            # No title (paper-friendly).
            ax.bar_label(bars, fmt="%.0f", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            # Highlight best (highest throughput is better)
            valid = [(i, v) for i, v in enumerate(values) if v and v > 0]
            if valid:
                best_idx = max(valid, key=lambda iv: iv[1])[0]
                bars[best_idx].set_edgecolor("gold")
                bars[best_idx].set_linewidth(3)
            panel_idx += 1

        # Memory panel
        if summary.kv_bytes_per_token:
            ax = axes[panel_idx]
            values = [summary.kv_bytes_per_token.get(m, 0) for m in model_names]
            bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1)
            ax.set_ylabel("KV-Cache Bytes/Token (↓ better)", fontsize=12)
            # No title (paper-friendly).
            ax.bar_label(bars, fmt="%.0f", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            # Highlight best (lowest memory is better)
            valid = [(i, v) for i, v in enumerate(values) if v and v > 0]
            if valid:
                best_idx = min(valid, key=lambda iv: iv[1])[0]
                bars[best_idx].set_edgecolor("gold")
                bars[best_idx].set_linewidth(3)

        fig.tight_layout()
        path = self.output_dir / "comparison_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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
        # No title (paper-friendly).
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = self.output_dir / "latency_vs_context.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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
        # No title (paper-friendly).
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = self.output_dir / "memory_vs_context.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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
        # No title (paper-friendly).
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

        fig.tight_layout()
        path = self.output_dir / "pareto_frontier.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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
                values_by_model[m].append((val / best) if (best and val) else 0)

        if summary.kv_bytes_per_token:
            metrics.append("Memory")
            best = min(summary.kv_bytes_per_token.values())
            for m in model_names:
                val = summary.kv_bytes_per_token.get(m)
                values_by_model[m].append((best / val) if (val and val > 0) else 0)

        if summary.efficiency_scores:
            metrics.append("Efficiency")
            best = max(summary.efficiency_scores.values())
            for m in model_names:
                val = summary.efficiency_scores.get(m)
                values_by_model[m].append((val / best) if (best and val) else 0)

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
        # No title (paper-friendly).

        fig.tight_layout()
        path = self.output_dir / "radar_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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

        fig, ax = plt.subplots(figsize=(max(8.0, len(model_names) * 1.5), len(metrics) * 1.2 + 1))

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

        # No title (paper-friendly).

        fig.tight_layout()
        path = self.output_dir / "metrics_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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

        fig, ax = plt.subplots(figsize=(max(8.0, len(model_names) * 2), len(rankings_data) * 0.8 + 1.5))
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

        # No title (paper-friendly).

        fig.tight_layout()
        path = self.output_dir / "rankings_table.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
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

        fig, ax = plt.subplots(figsize=(max(8.0, len(compare_models) * 2), 6))

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
        # No title (paper-friendly).
        ax.set_xticks(x)
        ax.set_xticklabels(compare_models, fontsize=11)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        path = self.output_dir / "deltas_from_baseline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close()
        return path

    def _plot_accuracy_per_task(
        self,
        accuracy_results: dict[str, AccuracyResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot individual downstream task scores."""
        # Collect all tasks
        all_tasks: set[str] = set()
        for r in accuracy_results.values():
            for t in r.tasks:
                all_tasks.add(t.task)

        tasks_sorted = sorted(all_tasks)
        if not tasks_sorted:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(tasks_sorted))
        width = 0.8 / len(model_names)

        for i, name in enumerate(model_names):
            if name in accuracy_results:
                r = accuracy_results[name]
                vals = []
                for t_name in tasks_sorted:
                    t_res = next((t for t in r.tasks if t.task == t_name), None)
                    vals.append(t_res.accuracy * 100 if t_res else 0)

                offset = (i - len(model_names) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=name, color=colors[i], alpha=0.8)

        ax.set_ylabel("Accuracy (%)", fontsize=12)
        # No title (paper-friendly).
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_sorted, rotation=30, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        path = self.output_dir / "accuracy_per_task.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_accuracy_radar(
        self,
        accuracy_results: dict[str, AccuracyResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Radar chart for downstream tasks."""
        # Collect all tasks
        all_tasks: set[str] = set()
        for r in accuracy_results.values():
            for t in r.tasks:
                all_tasks.add(t.task)

        tasks_sorted = sorted(all_tasks)
        if len(tasks_sorted) < 3:
            return None # Radar needs at least 3 axes

        angles = np.linspace(0, 2 * np.pi, len(tasks_sorted), endpoint=False).tolist()
        angles += angles[:1] # close the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, name in enumerate(model_names):
            if name in accuracy_results:
                r = accuracy_results[name]
                vals = []
                for t_name in tasks_sorted:
                    t_res = next((t for t in r.tasks if t.task == t_name), None)
                    vals.append(t_res.accuracy * 100 if t_res else 0)
                vals += vals[:1]

                ax.plot(angles, vals, color=colors[i], linewidth=2, label=name)
                ax.fill(angles, vals, color=colors[i], alpha=0.1)

        # Cast to any to avoid lint errors for polar-specific methods
        polar_ax: Any = ax
        polar_ax.set_theta_offset(np.pi / 2)
        polar_ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks_sorted, fontsize=10)
        ax.set_ylim(0, 100)
        # No title (paper-friendly).
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        path = self.output_dir / "accuracy_radar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_accuracy_heat_map(
        self,
        accuracy_results: dict[str, AccuracyResult],
        model_names: list[str],
    ) -> Path | None:
        """Plot a heatmap of accuracy across models and tasks."""
        all_tasks: set[str] = set()
        for r in accuracy_results.values():
            for t in r.tasks:
                all_tasks.add(t.task)

        tasks_sorted = sorted(all_tasks)
        if not tasks_sorted:
            return None

        # Prepare grid
        data = np.zeros((len(tasks_sorted), len(model_names)))
        for j, m_name in enumerate(model_names):
            if m_name in accuracy_results:
                r = accuracy_results[m_name]
                for i, t_name in enumerate(tasks_sorted):
                    t_res = next((t for t in r.tasks if t.task == t_name), None)
                    data[i, j] = t_res.accuracy if t_res else 0.0

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1)

        # Set labels
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(tasks_sorted)))
        ax.set_yticklabels(tasks_sorted)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

        # Annotate cells
        for i in range(len(tasks_sorted)):
            for j in range(len(model_names)):
                ax.text(j, i, f"{data[i, j]*100:.1f}%", ha="center", va="center", color="black" if data[i, j] < 0.7 else "white")

        # No title (paper-friendly).
        fig.tight_layout()

        path = self.output_dir / "accuracy_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_accuracy_timing(
        self,
        accuracy_results: dict[str, AccuracyResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot timing comparison for accuracy tasks.

        Shows how long each model takes to complete each task, demonstrating
        that compressed architectures are faster on real-world tasks.
        """
        # Collect all tasks
        all_tasks: set[str] = set()
        for r in accuracy_results.values():
            for t in r.tasks:
                if t.elapsed_seconds > 0:  # Only include tasks with timing data
                    all_tasks.add(t.task)

        tasks_sorted = sorted(all_tasks)
        if not tasks_sorted:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: Time per task (grouped bars)
        ax1 = axes[0]
        x = np.arange(len(tasks_sorted))
        width = 0.8 / len(model_names)

        for i, name in enumerate(model_names):
            if name in accuracy_results:
                r = accuracy_results[name]
                times = []
                for t_name in tasks_sorted:
                    t_res = next((t for t in r.tasks if t.task == t_name), None)
                    times.append(t_res.elapsed_seconds if t_res else 0)

                offset = (i - len(model_names) / 2 + 0.5) * width
                bars = ax1.bar(x + offset, times, width, label=name, color=colors[i], alpha=0.8)

                # Add time labels on bars
                for bar, t in zip(bars, times):
                    if t > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                f'{t:.1f}s', ha='center', va='bottom', fontsize=8, rotation=90)

        ax1.set_ylabel("Time (seconds)", fontsize=12)
        # No title (paper-friendly).
        ax1.set_xticks(x)
        ax1.set_xticklabels(tasks_sorted, rotation=30, ha="right")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend(loc="upper right")

        # Right panel: Examples per second (speed)
        ax2 = axes[1]

        for i, name in enumerate(model_names):
            if name in accuracy_results:
                r = accuracy_results[name]
                speeds = []
                for t_name in tasks_sorted:
                    t_res = next((t for t in r.tasks if t.task == t_name), None)
                    if t_res and t_res.elapsed_seconds > 0:
                        speeds.append(t_res.total / t_res.elapsed_seconds)
                    else:
                        speeds.append(0)

                offset = (i - len(model_names) / 2 + 0.5) * width
                bars = ax2.bar(x + offset, speeds, width, label=name, color=colors[i], alpha=0.8)

        ax2.set_ylabel("Examples per Second", fontsize=12)
        # No title (paper-friendly).
        ax2.set_xticks(x)
        ax2.set_xticklabels(tasks_sorted, rotation=30, ha="right")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(loc="upper right")

        # No suptitle (paper-friendly).
        fig.tight_layout()

        path = self.output_dir / "accuracy_timing.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_overall_comparison(
        self,
        perplexity_results: dict[str, PerplexityResult] | None,
        memory_results: dict[str, MemoryResult] | None,
        accuracy_results: dict[str, AccuracyResult] | None,
        latency_results: dict[str, LatencyResult] | None,
        behavioral_results: dict[str, dict] | None,
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot an overall radar/spider chart comparing models across all benchmarks.

        Normalizes each metric to 0-1 scale where higher is always better,
        then displays as a radar chart for easy visual comparison.
        """
        # Collect metrics for each model
        # Format: {metric_name: {model_name: value}}
        # All values normalized so higher = better
        metrics: dict[str, dict[str, float]] = {}

        # 1. Perplexity (lower is better -> invert)
        if perplexity_results:
            ppls = {name: r.perplexity for name, r in perplexity_results.items() if r.perplexity > 0}
            if ppls:
                max_ppl = max(ppls.values())
                metrics["Perplexity"] = {name: 1 - (ppl / max_ppl) for name, ppl in ppls.items()}

        # 2. Memory efficiency (lower is better -> invert)
        if memory_results:
            mems = {name: r.peak_memory_mb for name, r in memory_results.items() if r.peak_memory_mb > 0}
            if mems:
                max_mem = max(mems.values())
                metrics["Memory Eff."] = {name: 1 - (mem / max_mem) for name, mem in mems.items()}

        # 3. Average accuracy (higher is better)
        if accuracy_results:
            accs = {}
            for name, r in accuracy_results.items():
                if r.tasks:
                    avg_acc = sum(t.accuracy for t in r.tasks) / len(r.tasks)
                    accs[name] = avg_acc
            if accs:
                metrics["Accuracy"] = accs

        # 4. Throughput / Speed (higher is better) - from latency or accuracy timing
        if latency_results:
            speeds = {name: r.avg_tokens_per_second for name, r in latency_results.items() if r.avg_tokens_per_second > 0}
            if speeds:
                max_speed = max(speeds.values())
                metrics["Speed"] = {name: spd / max_speed for name, spd in speeds.items()}
        elif accuracy_results:
            # Use accuracy timing as proxy for speed
            speeds = {}
            for name, r in accuracy_results.items():
                total_examples = sum(t.total for t in r.tasks)
                total_time = sum(t.elapsed_seconds for t in r.tasks)
                if total_time > 0:
                    speeds[name] = total_examples / total_time
            if speeds:
                max_speed = max(speeds.values())
                metrics["Speed"] = {name: spd / max_speed for name, spd in speeds.items()}

        # 5. Behavioral score (higher is better)
        if behavioral_results:
            behav_scores = {}
            for name, r in behavioral_results.items():
                summary = r.get("summary", r)
                # Try various keys for accuracy
                score = summary.get("exact_match_rate",
                        summary.get("partial_or_better_rate",
                        summary.get("soft_accuracy", 0)))
                if score > 0:
                    behav_scores[name] = score
            if behav_scores:
                metrics["Behavioral"] = behav_scores

        # Filter to metrics that have data for at least 2 models
        metrics = {k: v for k, v in metrics.items() if len(v) >= 2}

        if len(metrics) < 2:
            return None  # Need at least 2 metrics for a meaningful radar

        metric_names = list(metrics.keys())
        num_metrics = len(metric_names)

        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for i, name in enumerate(model_names):
            values = []
            for metric in metric_names:
                values.append(metrics[metric].get(name, 0))
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=12)
        ax.set_ylim(0, 1.1)

        # Add gridlines at meaningful intervals
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color='gray')

        # No title (paper-friendly).
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        fig.tight_layout()

        path = self.output_dir / "overall_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_efficiency_frontier(
        self,
        accuracy_results: dict[str, AccuracyResult] | None,
        latency_results: dict[str, LatencyResult] | None,
        memory_results: dict[str, MemoryResult] | None,
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot accuracy vs speed/memory efficiency frontier.

        Shows which models achieve the best accuracy-efficiency trade-off.
        Models on the Pareto frontier (upper-right) are optimal - you can't
        improve one metric without sacrificing the other.
        """
        if not accuracy_results:
            return None

        # Compute average accuracy for each model
        accs: dict[str, float] = {}
        for name, r in accuracy_results.items():
            if r.tasks:
                accs[name] = sum(t.accuracy for t in r.tasks) / len(r.tasks)

        if len(accs) < 2:
            return None

        # Get speed data (prefer latency results, fall back to accuracy timing)
        speeds: dict[str, float] = {}
        if latency_results:
            for name, r in latency_results.items():
                if r.avg_tokens_per_second > 0:
                    speeds[name] = r.avg_tokens_per_second
        else:
            # Use accuracy timing as proxy
            for name, r in accuracy_results.items():
                total_examples = sum(t.total for t in r.tasks)
                total_time = sum(t.elapsed_seconds for t in r.tasks)
                if total_time > 0:
                    speeds[name] = total_examples / total_time

        # Get memory data
        mems: dict[str, float] = {}
        if memory_results:
            for name, r in memory_results.items():
                if r.peak_memory_mb > 0:
                    mems[name] = r.peak_memory_mb

        # Determine which efficiency metric to use
        has_speed = len(speeds) >= 2
        has_memory = len(mems) >= 2

        if not has_speed and not has_memory:
            return None

        # Create figure with 1 or 2 panels
        n_panels = (1 if has_speed else 0) + (1 if has_memory else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # Panel 1: Accuracy vs Speed
        if has_speed:
            ax = axes[panel_idx]
            panel_idx += 1

            for i, name in enumerate(model_names):
                if name in accs and name in speeds:
                    ax.scatter(speeds[name], accs[name] * 100,
                              s=200, c=colors[i], label=name,
                              edgecolors='black', linewidth=1.5, zorder=5)
                    # Add label near point
                    ax.annotate(name, (speeds[name], accs[name] * 100),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')

            # Draw Pareto frontier
            points = [(speeds[n], accs[n] * 100, n) for n in model_names
                      if n in accs and n in speeds]
            if len(points) >= 2:
                # Sort by speed, find Pareto-optimal points
                points_sorted = sorted(points, key=lambda p: p[0])
                pareto = [points_sorted[0]]
                for p in points_sorted[1:]:
                    if p[1] >= pareto[-1][1]:  # Higher accuracy
                        pareto.append(p)

                if len(pareto) >= 2:
                    pareto_x = [p[0] for p in pareto]
                    pareto_y = [p[1] for p in pareto]
                    ax.plot(pareto_x, pareto_y, 'g--', linewidth=2, alpha=0.7,
                           label='Pareto frontier', zorder=1)

            ax.set_xlabel("Speed (examples/sec or tok/s)", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            # No title (paper-friendly).
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')

            # Add "better" arrow indicator
            ax.annotate('', xy=(0.95, 0.95), xytext=(0.85, 0.85),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(0.92, 0.82, 'better', transform=ax.transAxes,
                   fontsize=10, color='green', style='italic')

        # Panel 2: Accuracy vs Memory Efficiency
        if has_memory:
            ax = axes[panel_idx]

            # Invert memory so higher = better (memory efficiency)
            max_mem = max(mems.values())
            mem_eff = {n: (max_mem - m) / max_mem * 100 for n, m in mems.items()}

            for i, name in enumerate(model_names):
                if name in accs and name in mem_eff:
                    ax.scatter(mem_eff[name], accs[name] * 100,
                              s=200, c=colors[i], label=name,
                              edgecolors='black', linewidth=1.5, zorder=5)
                    ax.annotate(name, (mem_eff[name], accs[name] * 100),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')

            # Draw Pareto frontier
            points = [(mem_eff[n], accs[n] * 100, n) for n in model_names
                      if n in accs and n in mem_eff]
            if len(points) >= 2:
                points_sorted = sorted(points, key=lambda p: p[0])
                pareto = [points_sorted[0]]
                for p in points_sorted[1:]:
                    if p[1] >= pareto[-1][1]:
                        pareto.append(p)

                if len(pareto) >= 2:
                    pareto_x = [p[0] for p in pareto]
                    pareto_y = [p[1] for p in pareto]
                    ax.plot(pareto_x, pareto_y, 'g--', linewidth=2, alpha=0.7,
                           label='Pareto frontier', zorder=1)

            ax.set_xlabel("Memory Efficiency (% savings vs max)", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            # No title (paper-friendly).
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')

            ax.annotate('', xy=(0.95, 0.95), xytext=(0.85, 0.85),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(0.92, 0.82, 'better', transform=ax.transAxes,
                   fontsize=10, color='green', style='italic')

        # No suptitle (paper-friendly).
        fig.tight_layout()

        path = self.output_dir / "efficiency_frontier.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_relative_improvement(
        self,
        perplexity_results: dict[str, PerplexityResult] | None,
        memory_results: dict[str, MemoryResult] | None,
        accuracy_results: dict[str, AccuracyResult] | None,
        latency_results: dict[str, LatencyResult] | None,
        model_names: list[str],
        colors: list[str],
        baseline_name: str | None = None,
    ) -> Path | None:
        """Plot relative improvement/regression vs baseline for each metric.

        Shows percentage change from baseline, making trade-offs immediately clear:
        e.g., "+15% speed, -2% accuracy, -25% memory"
        """
        # Determine baseline (first model if not specified)
        if baseline_name is None or baseline_name not in model_names:
            baseline_name = model_names[0] if model_names else None
        if baseline_name is None:
            return None

        # Collect baseline values
        baseline: dict[str, float] = {}

        def _kv_bytes_per_token(r: MemoryResult) -> float | None:
            ka = getattr(r, "kvcache_analysis", None)
            if ka is None:
                return None
            # Prefer DBA-specific bytes/tok when present (student/DBA models),
            # otherwise fall back to standard fp16 bytes/tok.
            v = getattr(ka, "bytes_per_token_dba_fp16", None)
            if v is not None and float(v) > 0:
                return float(v)
            v = getattr(ka, "bytes_per_token_fp16", None)
            return float(v) if v is not None and float(v) > 0 else None

        if perplexity_results and baseline_name in perplexity_results:
            baseline["Perplexity"] = perplexity_results[baseline_name].perplexity

        if memory_results and baseline_name in memory_results:
            # Use KV-cache bytes/token (the metric we report in memory_kv), not peak RSS.
            kv = _kv_bytes_per_token(memory_results[baseline_name])
            if kv is not None:
                baseline["KV bytes/tok"] = kv

        if accuracy_results and baseline_name in accuracy_results:
            r = accuracy_results[baseline_name]
            if r.tasks:
                baseline["Accuracy"] = sum(t.accuracy for t in r.tasks) / len(r.tasks)
            # Also get speed from timing
            total_time = sum(t.elapsed_seconds for t in r.tasks)
            total_examples = sum(t.total for t in r.tasks)
            if total_time > 0:
                baseline["Speed"] = total_examples / total_time

        if latency_results and baseline_name in latency_results:
            baseline["Speed"] = latency_results[baseline_name].avg_tokens_per_second

        if len(baseline) < 2:
            return None

        # Compute relative changes for other models
        other_models = [m for m in model_names if m != baseline_name]
        if not other_models:
            return None

        metrics = list(baseline.keys())

        # For each metric, define if higher is better
        higher_is_better = {
            "Perplexity": False,
            "KV bytes/tok": False,
            "Accuracy": True,
            "Speed": True,
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.8 / len(other_models)

        for i, name in enumerate(other_models):
            deltas = []
            for metric in metrics:
                base_val = baseline[metric]

                # Get this model's value
                model_val = None
                if metric == "Perplexity" and perplexity_results and name in perplexity_results:
                    model_val = perplexity_results[name].perplexity
                elif metric == "KV bytes/tok" and memory_results and name in memory_results:
                    model_val = _kv_bytes_per_token(memory_results[name])
                elif metric == "Accuracy" and accuracy_results and name in accuracy_results:
                    r = accuracy_results[name]
                    if r.tasks:
                        model_val = sum(t.accuracy for t in r.tasks) / len(r.tasks)
                elif metric == "Speed":
                    if latency_results and name in latency_results:
                        model_val = latency_results[name].avg_tokens_per_second
                    elif accuracy_results and name in accuracy_results:
                        r = accuracy_results[name]
                        total_time = sum(t.elapsed_seconds for t in r.tasks)
                        total_examples = sum(t.total for t in r.tasks)
                        if total_time > 0:
                            model_val = total_examples / total_time

                if model_val is not None and base_val > 0:
                    # Calculate % change
                    pct_change = ((model_val - base_val) / base_val) * 100
                    # Flip sign for "lower is better" metrics so positive = good
                    if not higher_is_better.get(metric, True):
                        pct_change = -pct_change
                    deltas.append(pct_change)
                else:
                    deltas.append(0)

            offset = (i - len(other_models) / 2 + 0.5) * width
            # Color bars: green for positive (improvement), red for negative
            bar_colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]
            bars = ax.bar(x + offset, deltas, width, label=name, color=colors[i + 1], alpha=0.8)

            # Add value labels
            for bar, d in zip(bars, deltas):
                label = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"
                y_pos = bar.get_height() + (1 if d >= 0 else -3)
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                       ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel("Change vs Baseline (%)", fontsize=12)
        # No title (paper-friendly).
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(loc='best')
        ax.grid(axis="y", alpha=0.3)

        # Add subtle background coloring
        ax.axhspan(0, ax.get_ylim()[1], alpha=0.05, color='green')
        ax.axhspan(ax.get_ylim()[0], 0, alpha=0.05, color='red')

        fig.tight_layout()

        path = self.output_dir / "relative_improvement.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        return path

    def _plot_context_sweep_chart(
        self,
        context_results: dict[str, ContextResult],
        model_names: list[str],
        colors: list[str],
    ) -> Path | None:
        """Plot throughput vs context length from context results."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(model_names):
            if name in context_results:
                r = context_results[name]
                # Filter successful runs
                measurements = [m for m in r.decode if m.ok]
                if not measurements:
                    continue

                x = [m.context_len for m in measurements]
                y = [m.decode_tok_per_s for m in measurements]

                # Sort by context length
                combined = sorted(zip(x, y))
                x_sorted, y_sorted = zip(*combined)

                ax.plot(x_sorted, y_sorted, "o-", color=colors[i], label=name, linewidth=2, markersize=8)

        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        ax.set_ylabel("Decode Throughput (tok/s)", fontsize=12)
        # No title (paper-friendly).
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = self.output_dir / "context_sweep.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
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
        behavioral_results: dict[str, dict] | None = None,
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

            # Loss row
            loss_vals = []
            for name in summary.model_names:
                if perplexity_results and name in perplexity_results:
                    loss_vals.append(f"{perplexity_results[name].loss:.4f}")
                else:
                    loss_vals.append("--")
            lines.append(f"PPL Loss & {' & '.join(loss_vals)} \\\\")

        # Throughput row
        if summary.throughputs:
            vals = [f"{summary.throughputs.get(m, 0):.0f}" for m in summary.model_names]
            best_idx = summary.throughput_rankings.index(summary.throughput_rankings[0]) if summary.throughput_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"Tokens/sec ($\\uparrow$) & {' & '.join(vals)} \\\\")

            # TTFT row
            ttft_vals = []
            prefill_vals = []
            for name in summary.model_names:
                if latency_results and name in latency_results:
                    ttft_vals.append(f"{latency_results[name].avg_time_to_first_token_ms:.1f}")
                    avg_prefill = np.mean([m.prefill_time_ms for m in latency_results[name].measurements]) if latency_results[name].measurements else 0
                    prefill_vals.append(f"{avg_prefill:.1f}")
                else:
                    ttft_vals.append("--")
                    prefill_vals.append("--")
            lines.append(f"TTFT (ms) & {' & '.join(ttft_vals)} \\\\")
            lines.append(f"Prefill (ms) & {' & '.join(prefill_vals)} \\\\")

        # Memory row
        if summary.kv_bytes_per_token:
            vals = [f"{summary.kv_bytes_per_token.get(m, 0):.0f}" for m in summary.model_names]
            best_idx = summary.memory_rankings.index(summary.memory_rankings[0]) if summary.memory_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"KV Bytes/tok ($\\downarrow$) & {' & '.join(vals)} \\\\")

            # Peak Memory row
            peak_vals = []
            for name in summary.model_names:
                if summary.peak_memory_mb and name in summary.peak_memory_mb:
                    peak_vals.append(f"{summary.peak_memory_mb[name]:.0f}")
                else:
                    peak_vals.append("--")
            lines.append(f"Peak Mem (MB) & {' & '.join(peak_vals)} \\\\")

        # Accuracy row
        if summary.micro_accuracies:
            vals = [f"{summary.micro_accuracies.get(m, 0) * 100:.1f}\\%" for m in summary.model_names]
            best_idx = summary.accuracy_rankings.index(summary.accuracy_rankings[0]) if summary.accuracy_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"Task Accuracy ($\\uparrow$) & {' & '.join(vals)} \\\\")

        # Behavioral row
        if summary.behavioral_exact_rates:
            vals = [f"{summary.behavioral_exact_rates.get(m, 0) * 100:.1f}\\%" for m in summary.model_names]
            best_idx = summary.behavioral_rankings.index(summary.behavioral_rankings[0]) if summary.behavioral_rankings else -1
            if best_idx >= 0:
                vals[best_idx] = f"\\textbf{{{vals[best_idx]}}}"
            lines.append(f"Behavioral (Exact) & {' & '.join(vals)} \\\\")

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
        # 7. PERPLEXITY TABLE (table_perplexity.tex)
        # ========================================================================
        if perplexity_results:
            ppl_lines = [
                "% Detailed perplexity results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Perplexity and Loss Comparison}",
                "\\label{tab:perplexity-detailed}",
                f"\\begin{{tabular}}{{l{'r' * 3}}}",
                "\\toprule",
                "Model & Perplexity & Loss & Tokens \\\\",
                "\\midrule",
            ]
            for name in summary.model_names:
                if name in perplexity_results:
                    r = perplexity_results[name]
                    ppl_lines.append(f"{name} & {r.perplexity:.2f} & {r.loss:.4f} & {r.num_tokens} \\\\")
                else:
                    ppl_lines.append(f"{name} & -- & -- & -- \\\\")
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
        # 8. LATENCY TABLE (table_latency.tex)
        # ========================================================================
        if latency_results:
            lat_lines = [
                "% Detailed latency results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Latency and Throughput Performance}",
                "\\label{tab:latency-detailed}",
                f"\\begin{{tabular}}{{l{'r' * 4}}}",
                "\\toprule",
                "Model & Tok/s & TTFT (ms) & Prefill (ms) & Decode (ms) \\\\",
                "\\midrule",
            ]
            for name in summary.model_names:
                if name in latency_results:
                    r = latency_results[name]
                    ttft = r.avg_time_to_first_token_ms
                    tps = r.avg_tokens_per_second
                    avg_prefill = np.mean([m.prefill_time_ms for m in r.measurements]) if r.measurements else 0
                    avg_decode = np.mean([m.decode_time_ms for m in r.measurements]) if r.measurements else 0
                    lat_lines.append(f"{name} & {tps:.1f} & {ttft:.1f} & {avg_prefill:.1f} & {avg_decode:.1f} \\\\")
                else:
                    lat_lines.append(f"{name} & -- & -- & -- & -- \\\\")
            lat_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_latency.tex"
            with open(path, "w") as f:
                f.write("\n".join(lat_lines))
            generated["table_latency.tex"] = path

        # ========================================================================
        # 9. MEMORY TABLE (table_memory.tex)
        # ========================================================================
        if memory_results:
            mem_lines = [
                "% Detailed memory results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Memory Usage and KV-Cache Efficiency}",
                "\\label{tab:memory-detailed}",
                f"\\begin{{tabular}}{{l{'r' * 3}}}",
                "\\toprule",
                "Model & Peak Mem (MB) & KV Bytes/tok & Reduction \\\\",
                "\\midrule",
            ]
            for name in summary.model_names:
                if name in memory_results:
                    r = memory_results[name]
                    kv_bytes = (
                        r.kvcache_analysis.bytes_per_token_dba_fp16
                        or r.kvcache_analysis.bytes_per_token_fp16
                    ) if r.kvcache_analysis else 0
                    reduction = summary.memory_reductions.get(name, 1.0) if summary.memory_reductions else 1.0
                    mem_lines.append(f"{name} & {r.peak_memory_mb:.0f} & {kv_bytes:.1f} & {reduction:.2f}x \\\\")
                else:
                    mem_lines.append(f"{name} & -- & -- & -- \\\\")
            mem_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_memory.tex"
            with open(path, "w") as f:
                f.write("\n".join(mem_lines))
            generated["table_memory.tex"] = path

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
                    all_ctx_lens.add(m.context_len)

            for ctx_len in sorted(all_ctx_lens):
                row_vals = []
                for name in summary.model_names:
                    if name in context_results:
                        r = context_results[name]
                        tps_vals = [m.decode_tok_per_s for m in r.decode if m.context_len == ctx_len and m.ok]
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

        # ========================================================================
        # 8. BEHAVIORAL DETAILED TABLE (table_behavioral.tex)
        # ========================================================================
        if behavioral_results:
            behav_lines = [
                "% Detailed behavioral evaluation results",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Behavioral V2 Match Quality}",
                "\\label{tab:behavioral-detailed}",
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "Model & Exact & Contained & None \\\\",
                "\\midrule",
            ]
            for name in summary.model_names:
                if name in behavioral_results:
                    r = behavioral_results[name]
                    # behavioral_results[name] contains 'summary' or direct keys depending on how it was stored
                    # in MultiModelBenchmarkRunner.run()
                    s = r.get("summary", r)
                    # Handle multiple naming conventions across different scorers
                    exact = s.get("exact_count", s.get("exact_match_count", s.get("exact_match_rate", 0)))
                    partial = s.get("contained_count", s.get("partial_count", s.get("partial_match_count", s.get("partial_match_rate", 0))))

                    # Handle both counts and rates
                    if isinstance(exact, float):
                        exact_str = f"{exact * 100:.1f}\\%"
                        partial_str = f"{partial * 100:.1f}\\%"
                        none_str = f"{(1 - exact - partial) * 100:.1f}\\%"
                    else:
                        exact_str = str(exact)
                        partial_str = str(partial)
                        # Handle both naming conventions
                        none_str = str(s.get("none_count", s.get("no_match_count", 0)))

                    behav_lines.append(f"{name} & {exact_str} & {partial_str} & {none_str} \\\\")

            behav_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            path = self.output_dir / "table_behavioral.tex"
            with open(path, "w") as f:
                f.write("\n".join(behav_lines))
            generated["table_behavioral.tex"] = path

        # ========================================================================
        # 9. BEHAVIORAL CATEGORY TABLE (table_behavioral_categories.tex)
        # ========================================================================
        # This uses the 'by_category' data if available
        has_category_data = any(
            name in behavioral_results and "by_category" in behavioral_results[name]
            for name in summary.model_names
        ) if behavioral_results else False

        if behavioral_results and has_category_data:
            # Collect all categories
            all_categories: set[str] = set()
            for name in summary.model_names:
                if name in behavioral_results and "by_category" in behavioral_results[name]:
                    all_categories.update(behavioral_results[name]["by_category"].keys())

            categories = sorted(all_categories)
            if categories:
                cat_lines = [
                    "% Per-category behavioral results",
                    "\\begin{table}[htbp]",
                    "\\centering",
                    "\\caption{Behavioral Accuracy by Category (Soft Match)}",
                    "\\label{tab:behavioral-categories}",
                    "\\small",
                    f"\\begin{{tabular}}{{l{'r' * len(categories)}}}",
                    "\\toprule",
                    "Model & " + " & ".join(categories) + " \\\\",
                    "\\midrule",
                ]
                for name in summary.model_names:
                    row_vals = []
                    if name in behavioral_results and "by_category" in behavioral_results[name]:
                        by_cat = behavioral_results[name]["by_category"]
                        for cat in categories:
                            cat_stats = by_cat.get(cat, {})
                            if isinstance(cat_stats, dict):
                                # Handle multiple naming conventions for accuracy
                                soft_acc = cat_stats.get("soft_accuracy",
                                           cat_stats.get("partial_or_better_rate",
                                           cat_stats.get("exact_rate",
                                           cat_stats.get("soft", 0.0))))
                            else:
                                try:
                                    soft_acc = float(cat_stats)
                                except (TypeError, ValueError):
                                    soft_acc = 0.0

                            if soft_acc is None:
                                soft_acc = 0.0
                            row_vals.append(f"{soft_acc * 100:.0f}\\%")
                    else:
                        row_vals = ["--"] * len(categories)
                    cat_lines.append(f"{name} & {' & '.join(row_vals)} \\\\")

                cat_lines.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}",
                ])
                path = self.output_dir / "table_behavioral_categories.tex"
                with open(path, "w") as f:
                    f.write("\n".join(cat_lines))
                generated["table_behavioral_categories.tex"] = path

        return generated
