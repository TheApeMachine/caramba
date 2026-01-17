"""Multi-model benchmark runner: orchestrating benchmarks across N models.

The runner is the entry point for multi-model benchmarking. It executes all
configured benchmarks on N models, then generates paper-ready artifacts
(CSV, JSON, PNG, LaTeX) with comprehensive comparisons.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from caramba.benchmark.artifacts import ExperimentMetadata
from caramba.benchmark.accuracy import BenchmarkAccuracy, AccuracyResult
from caramba.benchmark.behavior import BehaviorBenchmark, BehaviorMultiResult, dump_attention_multi_model
from caramba.benchmark.behavioral_v2 import BenchmarkBehavioralV2, BehavioralV2MultiResult
from caramba.benchmark.context import BenchmarkContext, ContextResult
from caramba.benchmark.latency import LatencyBenchmark, LatencyResult
from caramba.benchmark.memory import MemoryBenchmark, MemoryResult
from caramba.benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from caramba.config.benchmark import (
    AccuracyBenchmarkConfig,
    BehaviorBenchmarkConfig,
    BenchmarkSuite,
    BenchmarkType,
    BehavioralV2BenchmarkConfig,
    ContextBenchmarkConfig,
    LatencyBenchmarkConfig,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
)
from caramba.console import logger

if TYPE_CHECKING:
    from caramba.benchmark.multi_model_artifacts import MultiModelArtifactGenerator


class MultiModelBenchmarkRunner:
    """Runs all configured benchmarks across N models and generates artifacts.

    Orchestrates perplexity, latency, memory, accuracy, behavioral, and context
    benchmarks across multiple models to enable comprehensive N-way comparisons.
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        device: torch.device,
        metadata: ExperimentMetadata,
        baseline_name: str | None = None,
    ) -> None:
        """Set up the runner with benchmark suite and experiment metadata.

        Args:
            suite: Benchmark suite configuration.
            device: Device to run benchmarks on.
            metadata: Experiment metadata for artifact generation.
            baseline_name: Name of baseline model for delta calculations.
        """
        self.suite = suite
        self.device = device
        self.metadata = metadata
        self.baseline_name = baseline_name
        self.output_dir = Path(suite.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, models: dict[str, nn.Module]) -> dict[str, Path]:
        """Run all benchmarks on all models and generate artifacts.

        Args:
            models: Dict mapping model names to nn.Module instances.

        Returns:
            Dict mapping artifact names to their file paths.
        """
        model_names = sorted(models.keys())
        logger.header(
            "Multi-Model Benchmarks",
            f"{len(models)} models • {len(self.suite.benchmarks)} benchmarks",
        )
        logger.info(f"Models: {', '.join(model_names)}")
        if self.baseline_name:
            logger.info(f"Baseline: {self.baseline_name}")

        # Collect results by benchmark type, keyed by model name
        perplexity_results: dict[str, PerplexityResult] = {}
        latency_results: dict[str, LatencyResult] = {}
        memory_results: dict[str, MemoryResult] = {}
        accuracy_results: dict[str, AccuracyResult] = {}
        context_results: dict[str, ContextResult] = {}
        behavioral_v2_results: dict[str, dict] = {}
        behavior_multi_results: dict[str, BehaviorMultiResult] = {}

        for spec in self.suite.benchmarks:
            logger.subheader(f"{spec.id} ({spec.config.type})")

            # Determine which models to benchmark
            # If spec.models is set, only run those; otherwise run all
            models_to_run = spec.models if spec.models else model_names

            for _ in range(spec.repeats):
                match spec.config.type:
                    case BenchmarkType.PERPLEXITY:
                        assert isinstance(spec.config, PerplexityBenchmarkConfig)
                        benchmark = PerplexityBenchmark(spec.config, self.device)

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            result = benchmark.run(model, model_name)
                            # Keep best (lowest) perplexity
                            if (
                                model_name not in perplexity_results
                                or result.perplexity
                                < perplexity_results[model_name].perplexity
                            ):
                                perplexity_results[model_name] = result
                            logger.metric(model_name, result.perplexity, " ppl")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            result = benchmark.run(model, model_name)
                            # Keep first result (or could aggregate)
                            if model_name not in latency_results:
                                latency_results[model_name] = result
                            logger.metric(
                                model_name, result.avg_tokens_per_second, " tok/s"
                            )

                    case BenchmarkType.MEMORY:
                        assert isinstance(spec.config, MemoryBenchmarkConfig)
                        benchmark = MemoryBenchmark(spec.config, self.device)

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            result = benchmark.run(model, model_name)
                            if model_name not in memory_results:
                                memory_results[model_name] = result
                            if result.kvcache_analysis:
                                kv_bytes = (
                                    result.kvcache_analysis.bytes_per_token_dba_fp16
                                    or result.kvcache_analysis.bytes_per_token_fp16
                                )
                                logger.metric(model_name, kv_bytes, " bytes/tok")

                    case BenchmarkType.ACCURACY:
                        assert isinstance(spec.config, AccuracyBenchmarkConfig)
                        # Run all models per task before moving to next task
                        # This is better for comparison and easier to debug
                        accuracy_results = self._run_accuracy_task_first(
                            spec.config, models, models_to_run
                        )

                    case BenchmarkType.BEHAVIORAL_V2:
                        assert isinstance(spec.config, BehavioralV2BenchmarkConfig)
                        benchmark = BenchmarkBehavioralV2(spec.config, self.device)

                        # Filter models to run
                        run_models = {
                            name: models[name]
                            for name in models_to_run
                            if name in models
                        }

                        if run_models:
                            # Use the new N-model run_multi() method
                            result = benchmark.run_multi(
                                models=run_models,
                                output_dir=self.output_dir,
                                baseline_name=self.baseline_name,
                            )

                            # Store results for each model
                            for model_name, summary in result.model_summaries.items():
                                behavioral_v2_results[model_name] = {
                                    "exact_match_rate": summary.get("exact_match_rate", 0.0),
                                    "partial_or_better_rate": summary.get("partial_or_better_rate", 0.0),
                                    "summary": summary,
                                    "by_category": result.model_by_category.get(model_name, {}),
                                }
                                logger.metric(
                                    model_name,
                                    summary.get("exact_match_rate", 0.0) * 100.0,
                                    "% exact",
                                )
                        else:
                            logger.warning("No models available for behavioral v2")

                    case BenchmarkType.BEHAVIOR:
                        assert isinstance(spec.config, BehaviorBenchmarkConfig)
                        benchmark = BehaviorBenchmark(spec.config, self.device)

                        # Filter models to run
                        run_models = {
                            name: models[name]
                            for name in models_to_run
                            if name in models
                        }

                        if run_models:
                            # Use the new N-model run_multi() with weighted scoring
                            result = benchmark.run_multi(
                                models=run_models,
                                benchmark_id=spec.id,
                                output_dir=self.output_dir,
                                baseline_name=self.baseline_name,
                            )

                            # Store result for later artifact generation
                            behavior_multi_results[spec.id] = result

                            # Log weighted scores per model
                            for model_name in result.model_names:
                                hard_acc = result.get_hard_accuracy(model_name)
                                soft_acc = result.get_soft_accuracy(model_name)
                                weighted_acc = result.get_weighted_accuracy(model_name)
                                logger.metric(
                                    f"{model_name}_hard",
                                    hard_acc * 100.0,
                                    "%",
                                )
                                logger.metric(
                                    f"{model_name}_soft",
                                    soft_acc * 100.0,
                                    "%",
                                )
                                logger.metric(
                                    f"{model_name}_weighted",
                                    weighted_acc * 100.0,
                                    "%",
                                )
                        else:
                            logger.warning("No models available for behavior benchmark")

                    case BenchmarkType.CONTEXT:
                        assert isinstance(spec.config, ContextBenchmarkConfig)
                        benchmark = BenchmarkContext(spec.config, self.device)

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            result = benchmark.run(model, model_name)
                            context_results[model_name] = result
                            # Report last decode-tps at max context
                            try:
                                decode = result.decode
                                xs = [
                                    m.decode_tok_per_s
                                    for m in decode
                                    if m.ok
                                    and m.decode_tok_per_s == m.decode_tok_per_s
                                ]
                                if xs:
                                    logger.metric(model_name, float(xs[-1]), " tok/s@ctx")
                            except Exception as e:
                                logger.error(f"Failed to extract decode rate: {e!r}")

                    case _:
                        logger.warning(
                            f"Skipping unsupported benchmark type: {spec.config.type}"
                        )

        # Generate multi-model artifacts
        logger.info("Generating multi-model artifacts...")

        # Import here to avoid circular imports
        from caramba.benchmark.multi_model_artifacts import MultiModelArtifactGenerator

        generator = MultiModelArtifactGenerator(self.output_dir, self.baseline_name)
        paths = generator.generate_all(
            metadata=self.metadata,
            perplexity_results=perplexity_results,
            latency_results=latency_results,
            memory_results=memory_results,
            accuracy_results=accuracy_results,
            context_results=context_results,
            behavioral_results=behavioral_v2_results,
            formats=self.suite.formats,
        )

        logger.success(f"Generated {len(paths)} artifacts in {self.output_dir}")
        for name, path in paths.items():
            logger.path(str(path), name)

        return paths
