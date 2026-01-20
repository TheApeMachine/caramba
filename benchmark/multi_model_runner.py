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

from benchmark.artifacts import ExperimentMetadata
from benchmark.accuracy import BenchmarkAccuracy, AccuracyResult
from benchmark.behavior import BehaviorBenchmark, BehaviorMultiResult, dump_attention_multi_model
from benchmark.behavioral_v2 import BenchmarkBehavioralV2, BehavioralV2MultiResult
from benchmark.context import BenchmarkContext, ContextResult
from benchmark.latency import LatencyBenchmark, LatencyResult
from benchmark.memory import MemoryBenchmark, MemoryResult
from benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from config.benchmark import (
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
from console import logger

if TYPE_CHECKING:
    from benchmark.multi_model_artifacts import MultiModelArtifactGenerator


class MultiModelBenchmarkRunner:
    """Runs all configured benchmarks across N models and generates artifacts.

    Orchestrates perplexity, latency, memory, accuracy, behavioral, and context
    benchmarks across multiple models to enable comprehensive N-way comparisons.

    Artifacts are generated incrementally after each benchmark completes, so
    partial results are available even if the full run is interrupted.
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        device: torch.device,
        metadata: ExperimentMetadata,
        baseline_name: str | None = None,
        incremental_artifacts: bool = True,
    ) -> None:
        """Set up the runner with benchmark suite and experiment metadata.

        Args:
            suite: Benchmark suite configuration.
            device: Device to run benchmarks on.
            metadata: Experiment metadata for artifact generation.
            baseline_name: Name of baseline model for delta calculations.
            incremental_artifacts: If True, generate artifacts after each benchmark.
        """
        self.suite = suite
        self.device = device
        self.metadata = metadata
        self.baseline_name = baseline_name
        self.incremental_artifacts = incremental_artifacts
        self.output_dir = Path(suite.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated results for incremental artifact generation
        self._perplexity_results: dict[str, PerplexityResult] = {}
        self._latency_results: dict[str, LatencyResult] = {}
        self._memory_results: dict[str, MemoryResult] = {}
        self._accuracy_results: dict[str, AccuracyResult] = {}
        self._context_results: dict[str, ContextResult] = {}
        self._behavioral_v2_results: dict[str, dict] = {}
        self._behavior_multi_results: dict[str, BehaviorMultiResult] = {}

    def _save_incremental_artifacts(self, completed_benchmark: str) -> None:
        """Save artifacts incrementally after a benchmark completes.

        Args:
            completed_benchmark: Name of the benchmark that just completed.
        """
        if not self.incremental_artifacts:
            return

        try:
            from benchmark.multi_model_artifacts import MultiModelArtifactGenerator

            logger.info(f"Saving incremental artifacts after {completed_benchmark}...")

            generator = MultiModelArtifactGenerator(self.output_dir, self.baseline_name)

            # Combine behavioral results
            all_behavioral = self._behavioral_v2_results.copy()
            for spec_id, result in self._behavior_multi_results.items():
                ws = result.get_weighted_summary()
                models_data = ws.get("models", {})
                for model_name, stats in models_data.items():
                    if model_name not in all_behavioral:
                        all_behavioral[model_name] = {
                            "exact_match_rate": stats.get("hard_accuracy", 0.0),
                            "partial_or_better_rate": stats.get("soft_accuracy", 0.0),
                            "summary": stats,
                            "by_category": {},
                        }

            generator.generate_all(
                metadata=self.metadata,
                perplexity_results=self._perplexity_results or None,
                latency_results=self._latency_results or None,
                memory_results=self._memory_results or None,
                accuracy_results=self._accuracy_results or None,
                context_results=self._context_results or None,
                behavioral_results=all_behavioral or None,
                formats=self.suite.formats,
            )
            logger.info(f"Incremental artifacts saved to {self.output_dir}")

        except Exception as e:
            logger.warning(f"Failed to save incremental artifacts: {e!r}")

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

        # Clear accumulated results from any previous run
        self._perplexity_results.clear()
        self._latency_results.clear()
        self._memory_results.clear()
        self._accuracy_results.clear()
        self._context_results.clear()
        self._behavioral_v2_results.clear()
        self._behavior_multi_results.clear()

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
                                model_name not in self._perplexity_results
                                or result.perplexity
                                < self._perplexity_results[model_name].perplexity
                            ):
                                self._perplexity_results[model_name] = result
                            logger.metric(model_name, result.perplexity, " ppl")

                        # Save incremental artifacts after perplexity benchmark
                        self._save_incremental_artifacts("perplexity")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)
                        # Latency is resource-sensitive: isolate the active model on the accelerator.
                        cpu = torch.device("cpu")

                        def _gc_and_empty_cache() -> None:
                            try:
                                import gc

                                gc.collect()
                            except Exception:
                                pass
                            try:
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                elif self.device.type == "mps":
                                    torch.mps.empty_cache()
                            except Exception:
                                pass

                        def _try_move_model(model: nn.Module | None, device: torch.device) -> None:
                            if model is None:
                                return
                            try:
                                model.to(device)
                            except Exception:
                                pass

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            # Isolate: move other models to CPU, clear caches, ensure active model on device.
                            for other_name, other_model in models.items():
                                if other_name != model_name:
                                    _try_move_model(other_model, cpu)
                            _try_move_model(model, self.device)
                            _gc_and_empty_cache()
                            result = benchmark.run(model, model_name)
                            # Keep first result (or could aggregate)
                            if model_name not in self._latency_results:
                                self._latency_results[model_name] = result
                            logger.metric(
                                model_name, result.avg_tokens_per_second, " tok/s"
                            )

                        # Restore all models back to the benchmark device for subsequent benchmarks.
                        for _, m in models.items():
                            _try_move_model(m, self.device)
                        _gc_and_empty_cache()

                        # Save incremental artifacts after latency benchmark
                        self._save_incremental_artifacts("latency")

                    case BenchmarkType.MEMORY:
                        assert isinstance(spec.config, MemoryBenchmarkConfig)
                        benchmark = MemoryBenchmark(spec.config, self.device)
                        # Memory is extremely resource-sensitive: isolate the active model on the accelerator.
                        cpu = torch.device("cpu")

                        def _gc_and_empty_cache() -> None:
                            try:
                                import gc

                                gc.collect()
                            except Exception:
                                pass
                            try:
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                elif self.device.type == "mps":
                                    torch.mps.empty_cache()
                            except Exception:
                                pass

                        def _try_move_model(model: nn.Module | None, device: torch.device) -> None:
                            if model is None:
                                return
                            try:
                                model.to(device)
                            except Exception:
                                pass

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            # Isolate: move other models to CPU, clear caches, ensure active model on device.
                            for other_name, other_model in models.items():
                                if other_name != model_name:
                                    _try_move_model(other_model, cpu)
                            _try_move_model(model, self.device)
                            _gc_and_empty_cache()
                            result = benchmark.run(model, model_name)
                            if model_name not in self._memory_results:
                                self._memory_results[model_name] = result
                            if result.kvcache_analysis:
                                kv_bytes = (
                                    result.kvcache_analysis.bytes_per_token_dba_fp16
                                    or result.kvcache_analysis.bytes_per_token_fp16
                                )
                                logger.metric(model_name, kv_bytes, " bytes/tok")

                        # Restore all models back to the benchmark device for subsequent benchmarks.
                        for _, m in models.items():
                            _try_move_model(m, self.device)
                        _gc_and_empty_cache()

                        # Save incremental artifacts after memory benchmark
                        self._save_incremental_artifacts("memory")

                    case BenchmarkType.ACCURACY:
                        assert isinstance(spec.config, AccuracyBenchmarkConfig)
                        # Run all models per task before moving to next task
                        # This is better for comparison and easier to debug
                        self._accuracy_results = self._run_accuracy_task_first(
                            spec.config, models, models_to_run
                        )

                        # Save incremental artifacts after accuracy benchmark
                        self._save_incremental_artifacts("accuracy")

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
                                self._behavioral_v2_results[model_name] = {
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

                        # Save incremental artifacts after behavioral v2 benchmark
                        self._save_incremental_artifacts("behavioral_v2")

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
                            self._behavior_multi_results[spec.id] = result

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

                        # Save incremental artifacts after behavior benchmark
                        self._save_incremental_artifacts("behavior")

                    case BenchmarkType.CONTEXT:
                        assert isinstance(spec.config, ContextBenchmarkConfig)
                        benchmark = BenchmarkContext(spec.config, self.device)
                        cpu = torch.device("cpu")

                        def _gc_and_empty_cache() -> None:
                            try:
                                import gc

                                gc.collect()
                            except Exception:
                                pass
                            try:
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                elif self.device.type == "mps":
                                    torch.mps.empty_cache()
                            except Exception:
                                pass

                        def _try_move_model(model: nn.Module | None, device: torch.device) -> None:
                            if model is None:
                                return
                            try:
                                model.to(device)
                            except Exception:
                                pass

                        for model_name in models_to_run:
                            if model_name not in models:
                                logger.warning(
                                    f"Model '{model_name}' not found, skipping"
                                )
                                continue
                            model = models[model_name]
                            # Isolate the sweep on the accelerator: move other models to CPU.
                            for other_name, other_model in models.items():
                                if other_name != model_name:
                                    _try_move_model(other_model, cpu)
                            _try_move_model(model, self.device)
                            _gc_and_empty_cache()
                            result = benchmark.run(model, model_name)
                            self._context_results[model_name] = result
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

                        # Restore all models back to the benchmark device for subsequent benchmarks.
                        for _, m in models.items():
                            _try_move_model(m, self.device)
                        _gc_and_empty_cache()

                        # Save incremental artifacts after context benchmark
                        self._save_incremental_artifacts("context")

                    case _:
                        logger.warning(
                            f"Skipping unsupported benchmark type: {spec.config.type}"
                        )

        # Generate final multi-model artifacts
        logger.info("Generating final multi-model artifacts...")

        # Import here to avoid circular imports
        from benchmark.multi_model_artifacts import MultiModelArtifactGenerator

        generator = MultiModelArtifactGenerator(self.output_dir, self.baseline_name)
        # Combine all behavioral results (v2 and legacy)
        all_behavioral = self._behavioral_v2_results.copy()
        for spec_id, result in self._behavior_multi_results.items():
            ws = result.get_weighted_summary()
            models_data = ws.get("models", {})
            for model_name, stats in models_data.items():
                if model_name not in all_behavioral:
                    all_behavioral[model_name] = {
                        "exact_match_rate": stats.get("hard_accuracy", 0.0),
                        "partial_or_better_rate": stats.get("soft_accuracy", 0.0),
                        "summary": stats,
                        "by_category": {},  # Will be populated if available
                    }
                # Merge category data if present in legacy results
                # In behavior.py we now populate by_category in the WeightedModelSummary
                # but we need to extract it here if we want it in the main generator.
                # However, for now, we prioritize V2 categories.

        paths = generator.generate_all(
            metadata=self.metadata,
            perplexity_results=self._perplexity_results,
            latency_results=self._latency_results,
            memory_results=self._memory_results,
            accuracy_results=self._accuracy_results,
            context_results=self._context_results,
            behavioral_results=all_behavioral,
            formats=self.suite.formats,
        )

        logger.success(f"Generated {len(paths)} artifacts in {self.output_dir}")
        for name, path in paths.items():
            logger.path(str(path), name)

        return paths

    def _run_accuracy_task_first(
        self,
        config: AccuracyBenchmarkConfig,
        models: dict[str, nn.Module],
        models_to_run: list[str],
    ) -> dict[str, AccuracyResult]:
        """Run all models per task before moving to next task.

        OPTIMIZED: Reuses task dataset across models instead of rebuilding for each.

        Args:
            config: Accuracy benchmark configuration.
            models: Dict mapping model names to nn.Module instances.
            models_to_run: List of model names to run.

        Returns:
            Dict mapping model names to AccuracyResult instances.
        """
        from eval.logprob.completion.full_sequence import (
            LogprobCompletionFullSequence,
        )
        from eval.logprob.completion.windowed import (
            LogprobCompletionWindowed,
        )
        from eval.logprob.scorer import LogprobScorer
        from benchmark.accuracy.tasks.builder import (
            BenchmarkAccuracyTaskBuilder,
        )

        results: dict[str, AccuracyResult] = {
            name: AccuracyResult(model_name=name) for name in models_to_run
        }

        # Instantiate the benchmark to use its setup (tokenizer, etc.)
        benchmark = BenchmarkAccuracy(config, self.device)

        # Pre-compute context window setting once
        ctxw = (
            int(config.context_window)
            if config.context_window is not None
            else None
        )

        for task_name in list(config.tasks):
            t_name = str(task_name).strip().lower()
            if not t_name:
                continue

            # Subheader for the task itself
            logger.subheader(f"accuracy:{t_name}")

            # Build task ONCE per task (reuse across models)
            # We need a dummy scorer to build the task, then swap it per model
            first_model_name = next(
                (m for m in models_to_run if m in models), None
            )
            if first_model_name is None:
                logger.warning("No valid models to run")
                continue

            first_model = models[first_model_name]
            first_model.eval()

            # Create initial scorer with first model
            completion = (
                LogprobCompletionWindowed(
                    model=first_model, device=self.device, context_window=int(ctxw)
                )
                if ctxw is not None
                else LogprobCompletionFullSequence(model=first_model, device=self.device)
            )
            scorer = LogprobScorer(
                tokenizer=benchmark.tokenizer, completion=completion
            )

            builder = BenchmarkAccuracyTaskBuilder(
                scorer=scorer,
                config=config,
                output_dir=self.output_dir,
                device=self.device,
            )

            # Build task once - this loads dataset
            try:
                task = builder.build(t_name)
            except Exception as e:
                logger.error(f"Failed to build task '{t_name}': {e!r}")
                continue

            # Now run for each model, swapping the scorer
            for model_name in models_to_run:
                if model_name not in models:
                    logger.warning(f"Model '{model_name}' not found, skipping")
                    continue

                model = models[model_name]
                model.eval()

                # Create new completion scorer for this model (lightweight)
                completion = (
                    LogprobCompletionWindowed(
                        model=model, device=self.device, context_window=int(ctxw)
                    )
                    if ctxw is not None
                    else LogprobCompletionFullSequence(model=model, device=self.device)
                )
                # Swap scorer in task (reuses tokenizer)
                task.scorer = LogprobScorer(
                    tokenizer=benchmark.tokenizer, completion=completion
                )

                # Run the task
                try:
                    task_result = task.run(model=model, model_name=model_name)
                    results[model_name].tasks.append(task_result)

                    # Log the metric for this model/task
                    logger.metric(
                        f"{model_name}:{t_name}", task_result.accuracy * 100.0, "%"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to run task '{t_name}' for model '{model_name}': {e!r}"
                    )

        return results
