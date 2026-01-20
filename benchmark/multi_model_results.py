"""Multi-model comparison result types.

Dataclasses for holding and computing N-model comparison results,
including rankings, deltas from baseline, efficiency scores, and Pareto optimality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.perplexity import PerplexityResult
from benchmark.latency import LatencyResult
from benchmark.memory import MemoryResult
from benchmark.accuracy import AccuracyResult
from benchmark.context import ContextResult

import numpy as np


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model."""

    name: str
    perplexity: float | None = None
    tokens_per_second: float | None = None
    kv_bytes_per_token: float | None = None
    peak_memory_mb: float | None = None
    micro_accuracy: float | None = None
    behavioral_exact_rate: float | None = None
    behavioral_partial_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "perplexity": self.perplexity,
            "tokens_per_second": self.tokens_per_second,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "peak_memory_mb": self.peak_memory_mb,
            "micro_accuracy": self.micro_accuracy,
            "behavioral_exact_rate": self.behavioral_exact_rate,
            "behavioral_partial_rate": self.behavioral_partial_rate,
        }


@dataclass
class MultiModelComparisonSummary:
    """Comprehensive summary for N-model comparison."""

    model_names: list[str]
    baseline_name: str | None = None

    # Per-model metrics
    perplexities: dict[str, float] = field(default_factory=dict)
    throughputs: dict[str, float] = field(default_factory=dict)
    kv_bytes_per_token: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: dict[str, float] = field(default_factory=dict)
    micro_accuracies: dict[str, float] = field(default_factory=dict)
    behavioral_exact_rates: dict[str, float] = field(default_factory=dict)
    behavioral_partial_rates: dict[str, float] = field(default_factory=dict)

    # Rankings (best to worst)
    perplexity_rankings: list[str] = field(default_factory=list)
    throughput_rankings: list[str] = field(default_factory=list)
    memory_rankings: list[str] = field(default_factory=list)
    accuracy_rankings: list[str] = field(default_factory=list)
    behavioral_rankings: list[str] = field(default_factory=list)

    # Deltas from baseline
    perplexity_deltas: dict[str, float] | None = None
    speedups: dict[str, float] | None = None
    memory_reductions: dict[str, float] | None = None

    # Derived metrics
    efficiency_scores: dict[str, float] = field(default_factory=dict)
    pareto_optimal: list[str] = field(default_factory=list)

    def get_model_metrics(self, model_name: str) -> ModelMetrics:
        """Get aggregated metrics for a single model."""
        return ModelMetrics(
            name=model_name,
            perplexity=self.perplexities.get(model_name),
            tokens_per_second=self.throughputs.get(model_name),
            kv_bytes_per_token=self.kv_bytes_per_token.get(model_name),
            peak_memory_mb=self.peak_memory_mb.get(model_name),
            micro_accuracy=self.micro_accuracies.get(model_name),
            behavioral_exact_rate=self.behavioral_exact_rates.get(model_name),
            behavioral_partial_rate=self.behavioral_partial_rates.get(model_name),
        )

    def get_rank(self, model_name: str, metric: str) -> int | None:
        """Get rank for a model on a specific metric (1 = best)."""
        rankings_map = {
            "perplexity": self.perplexity_rankings,
            "throughput": self.throughput_rankings,
            "memory": self.memory_rankings,
            "accuracy": self.accuracy_rankings,
            "behavioral": self.behavioral_rankings,
        }
        rankings = rankings_map.get(metric, [])
        if model_name in rankings:
            return rankings.index(model_name) + 1
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_names": self.model_names,
            "baseline_name": self.baseline_name,
            "metrics": {
                "perplexities": self.perplexities,
                "throughputs": self.throughputs,
                "kv_bytes_per_token": self.kv_bytes_per_token,
                "peak_memory_mb": self.peak_memory_mb,
                "micro_accuracies": self.micro_accuracies,
                "behavioral_exact_rates": self.behavioral_exact_rates,
                "behavioral_partial_rates": self.behavioral_partial_rates,
            },
            "rankings": {
                "perplexity": self.perplexity_rankings,
                "throughput": self.throughput_rankings,
                "memory": self.memory_rankings,
                "accuracy": self.accuracy_rankings,
                "behavioral": self.behavioral_rankings,
            },
            "deltas": {
                "perplexity_deltas": self.perplexity_deltas,
                "speedups": self.speedups,
                "memory_reductions": self.memory_reductions,
            },
            "derived": {
                "efficiency_scores": self.efficiency_scores,
                "pareto_optimal": self.pareto_optimal,
            },
        }


def compute_rankings(values: dict[str, float], lower_is_better: bool = True) -> list[str]:
    """Compute rankings from a dict of values.

    Args:
        values: Dict mapping model names to metric values.
        lower_is_better: If True, lower values rank higher (e.g., perplexity).

    Returns:
        List of model names sorted from best to worst.
    """
    if not values:
        return []
    multiplier = 1 if lower_is_better else -1
    return sorted(values.keys(), key=lambda n: multiplier * values.get(n, float("inf")))


def compute_deltas(
    values: dict[str, float], baseline_name: str | None
) -> dict[str, float] | None:
    """Compute deltas from baseline.

    Args:
        values: Dict mapping model names to metric values.
        baseline_name: Name of baseline model.

    Returns:
        Dict mapping model names to delta from baseline, or None if no baseline.
    """
    if not baseline_name or baseline_name not in values:
        return None
    baseline_val = values[baseline_name]
    return {name: val - baseline_val for name, val in values.items()}


def compute_ratios(
    values: dict[str, float], baseline_name: str | None, invert: bool = False
) -> dict[str, float] | None:
    """Compute ratios relative to baseline.

    Args:
        values: Dict mapping model names to metric values.
        baseline_name: Name of baseline model.
        invert: If True, compute baseline/value (for memory reduction).
                If False, compute value/baseline (for speedup).

    Returns:
        Dict mapping model names to ratio vs baseline, or None if no baseline.
    """
    if not baseline_name or baseline_name not in values:
        return None
    baseline_val = values[baseline_name]
    if baseline_val == 0:
        return None
    if invert:
        return {name: baseline_val / val if val != 0 else 0 for name, val in values.items()}
    return {name: val / baseline_val for name, val in values.items()}


def compute_efficiency_scores(
    perplexities: dict[str, float],
    kv_bytes: dict[str, float],
    baseline_name: str | None,
) -> dict[str, float]:
    """Compute efficiency scores (quality/memory tradeoff).

    Higher score = better. Score of 1.0 for baseline (if present).
    Formula: (baseline_ppl / model_ppl) * (baseline_mem / model_mem)
    """
    if not perplexities or not kv_bytes:
        return {}

    # Use baseline or best values for normalization
    if baseline_name and baseline_name in perplexities and baseline_name in kv_bytes:
        base_ppl = perplexities[baseline_name]
        base_mem = kv_bytes[baseline_name]
    else:
        base_ppl = min(perplexities.values()) if perplexities else 1.0
        base_mem = min(kv_bytes.values()) if kv_bytes else 1.0

    scores = {}
    for name in set(perplexities.keys()) & set(kv_bytes.keys()):
        ppl = perplexities[name]
        mem = kv_bytes[name]
        if ppl > 0 and mem > 0:
            scores[name] = (base_ppl / ppl) * (base_mem / mem)
    return scores


def find_pareto_optimal(
    perplexities: dict[str, float],
    kv_bytes: dict[str, float],
) -> list[str]:
    """Find Pareto-optimal models (lower perplexity AND lower memory).

    A model is Pareto-optimal if no other model is strictly better on both axes.
    """
    models = [n for n in perplexities if n in kv_bytes]
    pareto = []

    for m1 in models:
        is_dominated = False
        for m2 in models:
            if m1 == m2:
                continue
            # m2 dominates m1 if m2 is <= on both axes and < on at least one
            if (
                perplexities[m2] <= perplexities[m1]
                and kv_bytes[m2] <= kv_bytes[m1]
            ):
                if perplexities[m2] < perplexities[m1] or kv_bytes[m2] < kv_bytes[m1]:
                    is_dominated = True
                    break
        if not is_dominated:
            pareto.append(m1)

    return pareto


def build_comparison_summary(
    *,
    perplexity_results: dict[str, PerplexityResult] | None = None,
    latency_results: dict[str, LatencyResult] | None = None,
    memory_results: dict[str, MemoryResult] | None = None,
    accuracy_results: dict[str, AccuracyResult] | None = None,
    context_results: dict[str, ContextResult] | None = None,
    behavioral_results: dict[str, dict] | None = None,
    baseline_name: str | None = None,
) -> MultiModelComparisonSummary:
    """Build comprehensive comparison summary from individual benchmark results.

    Args:
        perplexity_results: Dict mapping model names to PerplexityResult.
        latency_results: Dict mapping model names to LatencyResult.
        memory_results: Dict mapping model names to MemoryResult.
        accuracy_results: Dict mapping model names to AccuracyResult.
        context_results: Dict mapping model names to ContextResult.
        behavioral_results: Dict mapping model names to behavioral summary dicts.
        baseline_name: Name of baseline model for delta calculations.

    Returns:
        MultiModelComparisonSummary with all metrics, rankings, and derived values.
    """
    # Collect all model names
    all_models: set[str] = set()
    if perplexity_results:
        all_models.update(perplexity_results.keys())
    if latency_results:
        all_models.update(latency_results.keys())
    if memory_results:
        all_models.update(memory_results.keys())
    if accuracy_results:
        all_models.update(accuracy_results.keys())
    if behavioral_results:
        all_models.update(behavioral_results.keys())

    model_names = sorted(all_models)

    # Extract per-model metrics
    perplexities: dict[str, float] = {}
    if perplexity_results:
        perplexities = {name: r.perplexity for name, r in perplexity_results.items()}

    throughputs: dict[str, float] = {}
    if latency_results:
        throughputs = {
            name: r.avg_tokens_per_second for name, r in latency_results.items()
        }

    kv_bytes: dict[str, float] = {}
    peak_mem: dict[str, float] = {}
    if memory_results:
        for name, r in memory_results.items():
            if r.kvcache_analysis:
                kv_bytes[name] = (
                    r.kvcache_analysis.bytes_per_token_dba_fp16
                    or r.kvcache_analysis.bytes_per_token_fp16
                )
            peak_mem[name] = r.peak_memory_mb

    micro_accs: dict[str, float] = {}
    if accuracy_results:
        micro_accs = {name: r.micro_accuracy for name, r in accuracy_results.items()}

    behav_exact: dict[str, float] = {}
    behav_partial: dict[str, float] = {}
    if behavioral_results:
        for name, br in behavioral_results.items():
            if isinstance(br, dict):
                behav_exact[name] = br.get("exact_match_rate", 0.0)
                behav_partial[name] = br.get("partial_or_better_rate", 0.0)

    avg_ctx_tps: dict[str, float] = {}
    if context_results:
        for name, r in context_results.items():
            valid_decode = [m.decode_tok_per_s for m in r.decode if m.ok]
            if valid_decode:
                avg_ctx_tps[name] = float(np.mean(valid_decode))

    # Compute rankings
    ppl_rankings = compute_rankings(perplexities, lower_is_better=True)
    tps_rankings = compute_rankings(throughputs, lower_is_better=False)
    mem_rankings = compute_rankings(kv_bytes, lower_is_better=True)
    acc_rankings = compute_rankings(micro_accs, lower_is_better=False)
    behav_rankings = compute_rankings(behav_exact, lower_is_better=False)

    # Compute deltas and ratios
    ppl_deltas = compute_deltas(perplexities, baseline_name)
    speedups = compute_ratios(throughputs, baseline_name, invert=False)
    mem_reductions = compute_ratios(kv_bytes, baseline_name, invert=True)

    # Compute derived metrics
    efficiency = compute_efficiency_scores(perplexities, kv_bytes, baseline_name)
    pareto = find_pareto_optimal(perplexities, kv_bytes)

    return MultiModelComparisonSummary(
        model_names=model_names,
        baseline_name=baseline_name,
        perplexities=perplexities,
        throughputs=throughputs,
        kv_bytes_per_token=kv_bytes,
        peak_memory_mb=peak_mem,
        micro_accuracies=micro_accs,
        behavioral_exact_rates=behav_exact,
        behavioral_partial_rates=behav_partial,
        perplexity_rankings=ppl_rankings,
        throughput_rankings=tps_rankings,
        memory_rankings=mem_rankings,
        accuracy_rankings=acc_rankings,
        behavioral_rankings=behav_rankings,
        perplexity_deltas=ppl_deltas,
        speedups=speedups,
        memory_reductions=mem_reductions,
        efficiency_scores=efficiency,
        pareto_optimal=pareto,
    )
