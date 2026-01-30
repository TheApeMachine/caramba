"""Multi-model benchmark runner: orchestrating benchmarks across N models.

The runner is the entry point for multi-model benchmarking. It executes all
configured benchmarks on N models, then generates paper-ready artifacts
(CSV, JSON, PNG, LaTeX) with comprehensive comparisons.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from benchmark.artifacts import ExperimentMetadata
from benchmark.accuracy import BenchmarkAccuracy, AccuracyResult
from benchmark.behavior import BehaviorBenchmark, BehaviorResult
from benchmark.behavior_instruct import BenchmarkBehaviorInstruct
from benchmark.context import BenchmarkContext, ContextResult
from benchmark.latency import LatencyBenchmark, LatencyResult
from benchmark.memory import MemoryBenchmark, MemoryResult
from benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from benchmark.generation import GenerationBenchmark
from benchmark.utils import stitch_images, with_plotter
from config.benchmark import (
    AccuracyBenchmarkConfig,
    BehaviorBenchmarkConfig,
    BehaviorInstructBenchmarkConfig,
    BenchmarkSuite,
    BenchmarkType,
    ContextBenchmarkConfig,
    LatencyBenchmarkConfig,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
    GenerationBenchmarkConfig,
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
        self._behavior_results: dict[str, dict] = {}
        self._behavior_runs: dict[str, BehaviorResult] = {}
        # Full audit trail to embed in report.json.
        self._audit: dict[str, object] = {}

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

            generator.generate_all(
                metadata=self.metadata,
                perplexity_results=self._perplexity_results or None,
                latency_results=self._latency_results or None,
                memory_results=self._memory_results or None,
                accuracy_results=self._accuracy_results or None,
                context_results=self._context_results or None,
                behavioral_results=self._behavior_results or None,
                audit=self._audit or None,
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

        self._audit = {
            "suite": self.suite.model_dump(),
            "device": str(self.device),
            "baseline_name": self.baseline_name,
            "verifications": [],
        }

        # Clear accumulated results from any previous run
        self._perplexity_results.clear()
        self._latency_results.clear()
        self._memory_results.clear()
        self._accuracy_results.clear()
        self._context_results.clear()
        self._behavior_results.clear()
        self._behavior_runs.clear()

        raw_plot_paths: list[Path] = []

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

                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_perplexity_raw.png"
                        with with_plotter(
                            title=f"Perplexity raw loss • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for model_name in models_to_run:
                                if model_name not in models:
                                    raise RuntimeError(f"Model '{model_name}' not found (requested by spec '{spec.id}').")
                                model = models[model_name]
                                result = benchmark.run(
                                    model,
                                    model_name,
                                    plotter=(plotter if getattr(spec, "realtime", False) else None),  # type: ignore[arg-type]
                                    plot_series=str(model_name),
                                )
                                # FAIR aggregation across repeats: token-weighted average loss.
                                prev = self._perplexity_results.get(model_name)
                                if prev is None:
                                    self._perplexity_results[model_name] = result
                                else:
                                    tot_tokens = int(prev.num_tokens) + int(result.num_tokens)
                                    if tot_tokens <= 0:
                                        raise ValueError(f"Perplexity repeats produced no tokens for model '{model_name}'.")
                                    tot_loss = float(prev.loss) * float(prev.num_tokens) + float(result.loss) * float(result.num_tokens)
                                    avg_loss = tot_loss / float(tot_tokens)
                                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                                    self._perplexity_results[model_name] = PerplexityResult(
                                        model_name=str(model_name),
                                        perplexity=float(ppl),
                                        loss=float(avg_loss),
                                        num_tokens=int(tot_tokens),
                                        num_batches=int(prev.num_batches) + int(result.num_batches),
                                        batch_loss_sums=list(getattr(prev, "batch_loss_sums", [])) + list(getattr(result, "batch_loss_sums", [])),
                                        batch_token_counts=list(getattr(prev, "batch_token_counts", [])) + list(getattr(result, "batch_token_counts", [])),
                                    )
                                logger.metric(model_name, result.perplexity, " ppl")
                            if getattr(spec, "realtime", False):
                                try:
                                    plotter.save(raw_path)  # type: ignore[union-attr]
                                    raw_plot_paths.append(raw_path)
                                except Exception:
                                    pass

                        # Save incremental artifacts after perplexity benchmark
                        self._save_incremental_artifacts("perplexity")

                    case BenchmarkType.GENERATION:
                        assert isinstance(spec.config, GenerationBenchmarkConfig)
                        benchmark = GenerationBenchmark(spec.config, self.device)
                        for model_name in models_to_run:
                            if model_name not in models:
                                raise RuntimeError(f"Model '{model_name}' not found (requested by spec '{spec.id}').")
                            model = models[model_name]
                            benchmark.run(model, str(model_name), output_dir=self.output_dir)
                        self._save_incremental_artifacts("generation")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)
                        # Latency is resource-sensitive: isolate the active model on the accelerator.
                        cpu = torch.device("cpu")

                        def _gc_and_empty_cache() -> None:
                            import gc

                            gc.collect()
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                            elif self.device.type == "mps":
                                torch.mps.empty_cache()

                        def _try_move_model(model: nn.Module | None, device: torch.device) -> None:
                            if model is None:
                                return
                            model.to(device)

                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_latency_raw.png"
                        # Live plot: tokens/s over measurements (by model)
                        with with_plotter(
                            title=f"Latency raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for model_name in models_to_run:
                                if model_name not in models:
                                    raise RuntimeError(f"Model '{model_name}' not found (requested by spec '{spec.id}').")
                                model = models[model_name]
                                # Isolate: move other models to CPU, clear caches, ensure active model on device.
                                for other_name, other_model in models.items():
                                    if other_name != model_name:
                                        _try_move_model(other_model, cpu)
                                _try_move_model(model, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(
                                    model,
                                    model_name,
                                    on_measurement=(
                                        (
                                            lambda m, mn=str(model_name): plotter.log(
                                                **{f"{mn}/tok_s": float(m.tokens_per_second)}
                                            )
                                        )
                                        if getattr(spec, "realtime", False)
                                        else None
                                    ),
                                )
                                # FAIR aggregation across repeats: keep all measurements.
                                prev = self._latency_results.get(model_name)
                                if prev is None:
                                    self._latency_results[model_name] = result
                                else:
                                    prev.measurements.extend(result.measurements)
                                logger.metric(
                                    model_name, result.avg_tokens_per_second, " tok/s"
                                )
                                if getattr(spec, "realtime", False):
                                    plotter.log(  # type: ignore[union-attr]
                                        **{f"{model_name}/ttft_ms": float(result.avg_time_to_first_token_ms)}
                                    )

                            if getattr(spec, "realtime", False):
                                plotter.save(raw_path)  # type: ignore[union-attr]
                                raw_plot_paths.append(raw_path)

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
                            import gc

                            gc.collect()
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                            elif self.device.type == "mps":
                                torch.mps.empty_cache()

                        def _try_move_model(model: nn.Module | None, device: torch.device) -> None:
                            if model is None:
                                return
                            model.to(device)

                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_memory_raw.png"
                        with with_plotter(
                            title=f"Memory raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for model_name in models_to_run:
                                if model_name not in models:
                                    raise RuntimeError(f"Model '{model_name}' not found (requested by spec '{spec.id}').")
                                model = models[model_name]
                                # Isolate: move other models to CPU, clear caches, ensure active model on device.
                                for other_name, other_model in models.items():
                                    if other_name != model_name:
                                        _try_move_model(other_model, cpu)
                                _try_move_model(model, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(
                                    model,
                                    model_name,
                                    on_measurement=(
                                        (
                                            lambda m, mn=str(model_name): plotter.log(
                                                **{f"{mn}/kv_mb": float(m.kvcache_memory_mb)}
                                            )
                                        )
                                        if getattr(spec, "realtime", False)
                                        else None
                                    ),
                                )
                                prev = self._memory_results.get(model_name)
                                if prev is None:
                                    self._memory_results[model_name] = result
                                else:
                                    if prev.kvcache_analysis != result.kvcache_analysis:
                                        raise ValueError(
                                            f"Memory repeats produced inconsistent kvcache_analysis for '{model_name}'."
                                        )
                                    prev.measurements.extend(result.measurements)
                                if result.kvcache_analysis:
                                    kv_bytes = (
                                        result.kvcache_analysis.bytes_per_token_dba_fp16
                                        or result.kvcache_analysis.bytes_per_token_fp16
                                    )
                                    logger.metric(model_name, kv_bytes, " bytes/tok")
                                if getattr(spec, "realtime", False):
                                    plotter.log(  # type: ignore[union-attr]
                                        **{f"{model_name}/peak_mb": float(result.peak_memory_mb)}
                                    )

                            if getattr(spec, "realtime", False):
                                plotter.save(raw_path)  # type: ignore[union-attr]
                                raw_plot_paths.append(raw_path)

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
                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_accuracy_raw.png"
                        with with_plotter(
                            title=f"Accuracy raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            self._accuracy_results = self._run_accuracy_task_first(
                                spec.config,
                                models,
                                models_to_run,
                                plotter=(plotter if getattr(spec, "realtime", False) else None),  # type: ignore[arg-type]
                            )
                            if getattr(spec, "realtime", False):
                                try:
                                    plotter.save(raw_path)  # type: ignore[union-attr]
                                    raw_plot_paths.append(raw_path)
                                except Exception:
                                    pass

                        # Save incremental artifacts after accuracy benchmark
                        self._save_incremental_artifacts("accuracy")

                    case BenchmarkType.BEHAVIOR_INSTRUCT:
                        assert isinstance(spec.config, BehaviorInstructBenchmarkConfig)
                        benchmark = BenchmarkBehaviorInstruct(spec.config, self.device)

                        run_models = {
                            name: models[name]
                            for name in models_to_run
                            if name in models
                        }

                        if run_models:
                            result = benchmark.run_multi(models=run_models, output_dir=self.output_dir)
                            for model_name, summary in result.model_summaries.items():
                                self._behavior_results[str(model_name)] = {
                                    "exact_match_rate": summary.get("exact_match_rate", 0.0),
                                    "partial_or_better_rate": summary.get("partial_or_better_rate", 0.0),
                                    "summary": summary,
                                    "by_category": result.model_by_category.get(model_name, {}),
                                }
                        else:
                            logger.warning("No models available for behavior_instruct")

                        # Save incremental artifacts after behavior_instruct benchmark
                        self._save_incremental_artifacts("behavior_instruct")

                    case BenchmarkType.BEHAVIOR:
                        assert isinstance(spec.config, BehaviorBenchmarkConfig)
                        run_models = {name: models[name] for name in models_to_run if name in models}
                        if not run_models:
                            raise RuntimeError(f"{spec.id}: no models available for behavior benchmark.")
                        if not self._perplexity_results:
                            raise RuntimeError(
                                f"{spec.id}: behavior requires perplexity results to be available "
                                f"(place perplexity benchmark before behavior in the suite)."
                            )
                        ppl_by_model = {n: float(self._perplexity_results[n].perplexity) for n in run_models.keys()}
                        benchmark = BehaviorBenchmark(
                            suite_file=str(spec.config.suite_file),
                            tokenizer_config=spec.config.tokenizer,
                            device=self.device,
                        )
                        result = benchmark.run_multi(
                            models=run_models,
                            benchmark_id=str(spec.id),
                            output_dir=self.output_dir,
                            baseline_name=self.baseline_name,
                            seed=int(spec.config.seed),
                            max_new_tokens=int(spec.config.max_new_tokens),
                            context_window=spec.config.context_window,
                            dump_attention=bool(spec.config.dump_attention),
                            dump_attention_max_tokens=int(spec.config.dump_attention_max_tokens),
                            dump_attention_max_heads=int(spec.config.dump_attention_max_heads),
                            dump_attention_anchor=str(spec.config.dump_attention_anchor),
                            ppl_by_model=ppl_by_model,
                        )
                        self._behavior_runs[str(spec.id)] = result
                        self._behavior_results.update(_to_behavioral_results_dict(result))
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
                            model.to(device)

                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_context_raw.png"
                        with with_plotter(
                            title=f"Context raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for model_name in models_to_run:
                                if model_name not in models:
                                    raise RuntimeError(f"Model '{model_name}' not found (requested by spec '{spec.id}').")
                                model = models[model_name]
                                # Isolate the sweep on the accelerator: move other models to CPU.
                                for other_name, other_model in models.items():
                                    if other_name != model_name:
                                        _try_move_model(other_model, cpu)
                                _try_move_model(model, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(
                                    model,
                                    model_name,
                                    on_step=(
                                        (
                                            lambda r, mn=str(model_name): plotter.log(
                                                **{
                                                    f"{mn}/ppl": float(r.sweep[-1].ppl)
                                                    if r.sweep
                                                    else float("nan"),
                                                    f"{mn}/decode_tok_s": float(
                                                        r.decode[-1].decode_tok_per_s
                                                    )
                                                    if r.decode
                                                    else float("nan"),
                                                }
                                            )
                                        )
                                        if getattr(spec, "realtime", False)
                                        else None
                                    ),
                                )
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
                                        logger.metric(
                                            model_name, float(xs[-1]), " tok/s@ctx"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to extract decode rate: {e!r}"
                                    )

                            if getattr(spec, "realtime", False):
                                try:
                                    plotter.save(raw_path)  # type: ignore[union-attr]
                                    raw_plot_paths.append(raw_path)
                                except Exception:
                                    pass

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

        paths = self._generate_final_artifacts()
        # Stitch all "raw" benchmark plots into one comparison image (if any).
        try:
            if raw_plot_paths:
                out_path = self.output_dir / "raw" / "benchmarks_raw_comparison.png"
                stitched = stitch_images(
                    images=raw_plot_paths,
                    out_path=out_path,
                    cols=2,
                    title="Raw benchmark plots (realtime)",
                )
                if stitched is not None:
                    paths["raw_benchmarks_comparison"] = stitched
        except Exception:
            pass
        return paths

    def run_isolated(
        self,
        model_names: list[str],
        load_model,
        unload_model,
    ) -> dict[str, Path]:
        """Run all benchmarks, loading/unloading one model at a time.

        This reduces cross-model interference (allocator pressure, MPS fragmentation,
        caches) at the cost of extra load time.
        """
        logger.header(
            "Multi-Model Benchmarks (isolated)",
            f"{len(model_names)} models • {len(self.suite.benchmarks)} benchmarks",
        )
        logger.info(f"Models: {', '.join(model_names)}")
        if self.baseline_name:
            logger.info(f"Baseline: {self.baseline_name}")

        self._perplexity_results.clear()
        self._latency_results.clear()
        self._memory_results.clear()
        self._accuracy_results.clear()
        self._context_results.clear()
        self._behavior_results.clear()
        self._behavior_runs.clear()

        def _with_model(name: str):
            m = load_model(name)
            try:
                return m
            except Exception:
                unload_model(m)
                raise

        raw_plot_paths: list[Path] = []

        for spec in self.suite.benchmarks:
            logger.subheader(f"{spec.id} ({spec.config.type})")
            models_to_run = spec.models if spec.models else list(model_names)

            for _ in range(spec.repeats):
                match spec.config.type:
                    case BenchmarkType.PERPLEXITY:
                        assert isinstance(spec.config, PerplexityBenchmarkConfig)
                        benchmark = PerplexityBenchmark(spec.config, self.device)
                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_perplexity_raw.png"
                        with with_plotter(
                            title=f"Perplexity raw loss • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for name in models_to_run:
                                if name not in model_names:
                                    raise RuntimeError(f"Model '{name}' not found (requested by spec '{spec.id}').")
                                m = _with_model(name)
                                try:
                                    result = benchmark.run(
                                        m,
                                        name,
                                        plotter=(plotter if getattr(spec, "realtime", False) else None),  # type: ignore[arg-type]
                                        plot_series=str(name),
                                    )
                                    prev = self._perplexity_results.get(name)
                                    if prev is None:
                                        self._perplexity_results[name] = result
                                    else:
                                        tot_tokens = int(prev.num_tokens) + int(result.num_tokens)
                                        if tot_tokens <= 0:
                                            raise ValueError(f"Perplexity repeats produced no tokens for model '{name}'.")
                                        tot_loss = float(prev.loss) * float(prev.num_tokens) + float(result.loss) * float(result.num_tokens)
                                        avg_loss = tot_loss / float(tot_tokens)
                                        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                                        self._perplexity_results[name] = PerplexityResult(
                                            model_name=str(name),
                                            perplexity=float(ppl),
                                            loss=float(avg_loss),
                                            num_tokens=int(tot_tokens),
                                            num_batches=int(prev.num_batches) + int(result.num_batches),
                                            batch_loss_sums=list(getattr(prev, "batch_loss_sums", [])) + list(getattr(result, "batch_loss_sums", [])),
                                            batch_token_counts=list(getattr(prev, "batch_token_counts", [])) + list(getattr(result, "batch_token_counts", [])),
                                        )
                                    logger.metric(name, result.perplexity, " ppl")
                                finally:
                                    unload_model(m)
                            if getattr(spec, "realtime", False):
                                try:
                                    plotter.save(raw_path)  # type: ignore[union-attr]
                                    raw_plot_paths.append(raw_path)
                                except Exception:
                                    pass
                        self._save_incremental_artifacts("perplexity")

                    case BenchmarkType.GENERATION:
                        assert isinstance(spec.config, GenerationBenchmarkConfig)
                        benchmark = GenerationBenchmark(spec.config, self.device)
                        for name in models_to_run:
                            if name not in model_names:
                                continue
                            m = _with_model(name)
                            try:
                                benchmark.run(m, str(name), output_dir=self.output_dir)
                            finally:
                                unload_model(m)
                        self._save_incremental_artifacts("generation")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)
                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_latency_raw.png"
                        with with_plotter(
                            title=f"Latency raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for name in models_to_run:
                                if name not in model_names:
                                    continue
                                m = _with_model(name)
                                try:
                                    result = benchmark.run(
                                        m,
                                        name,
                                        on_measurement=(
                                            (
                                                lambda mm, mn=str(name): plotter.log(
                                                    **{f"{mn}/tok_s": float(mm.tokens_per_second)}
                                                )
                                            )
                                            if getattr(spec, "realtime", False)
                                            else None
                                        ),
                                    )
                                    prev = self._latency_results.get(name)
                                    if prev is None:
                                        self._latency_results[name] = result
                                    else:
                                        prev.measurements.extend(result.measurements)
                                    logger.metric(name, result.avg_tokens_per_second, " tok/s")
                                finally:
                                    unload_model(m)
                            if getattr(spec, "realtime", False):
                                plotter.save(raw_path)  # type: ignore[union-attr]
                                raw_plot_paths.append(raw_path)
                        self._save_incremental_artifacts("latency")

                    case BenchmarkType.MEMORY:
                        assert isinstance(spec.config, MemoryBenchmarkConfig)
                        benchmark = MemoryBenchmark(spec.config, self.device)
                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_memory_raw.png"
                        with with_plotter(
                            title=f"Memory raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for name in models_to_run:
                                if name not in model_names:
                                    raise RuntimeError(f"Model '{name}' not found (requested by spec '{spec.id}').")
                                m = _with_model(name)
                                try:
                                    result = benchmark.run(
                                        m,
                                        name,
                                        on_measurement=(
                                            (
                                                lambda mm, mn=str(name): plotter.log(
                                                    **{f"{mn}/kv_mb": float(mm.kvcache_memory_mb)}
                                                )
                                            )
                                            if getattr(spec, "realtime", False)
                                            else None
                                        ),
                                    )
                                    prev = self._memory_results.get(name)
                                    if prev is None:
                                        self._memory_results[name] = result
                                    else:
                                        if prev.kvcache_analysis != result.kvcache_analysis:
                                            raise ValueError(
                                                f"Memory repeats produced inconsistent kvcache_analysis for '{name}'."
                                            )
                                        prev.measurements.extend(result.measurements)
                                finally:
                                    unload_model(m)
                            if getattr(spec, "realtime", False):
                                plotter.save(raw_path)  # type: ignore[union-attr]
                                raw_plot_paths.append(raw_path)
                        self._save_incremental_artifacts("memory")

                    case BenchmarkType.ACCURACY:
                        assert isinstance(spec.config, AccuracyBenchmarkConfig)
                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_accuracy_raw.png"
                        with with_plotter(
                            title=f"Accuracy raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            self._accuracy_results = self._run_accuracy_task_first_isolated(
                                spec.config,
                                models_to_run,
                                load_model,
                                unload_model,
                                plotter=(plotter if getattr(spec, "realtime", False) else None),  # type: ignore[arg-type]
                            )
                            if getattr(spec, "realtime", False):
                                try:
                                    plotter.save(raw_path)  # type: ignore[union-attr]
                                    raw_plot_paths.append(raw_path)
                                except Exception:
                                    pass
                        self._save_incremental_artifacts("accuracy")

                    case BenchmarkType.BEHAVIOR_INSTRUCT:
                        assert isinstance(spec.config, BehaviorInstructBenchmarkConfig)
                        benchmark = BenchmarkBehaviorInstruct(spec.config, self.device)
                        result = benchmark.run_multi_isolated(
                            model_names=[n for n in models_to_run if n in model_names],
                            load_model=load_model,
                            unload_model=unload_model,
                            output_dir=self.output_dir,
                        )
                        for model_name, summary in result.model_summaries.items():
                            self._behavior_results[str(model_name)] = {
                                "exact_match_rate": summary.get("exact_match_rate", 0.0),
                                "partial_or_better_rate": summary.get("partial_or_better_rate", 0.0),
                                "summary": summary,
                                "by_category": result.model_by_category.get(model_name, {}),
                            }
                        self._save_incremental_artifacts("behavior_instruct")

                    case BenchmarkType.BEHAVIOR:
                        assert isinstance(spec.config, BehaviorBenchmarkConfig)
                        run_names = [n for n in models_to_run if n in model_names]
                        if not run_names:
                            raise RuntimeError(f"{spec.id}: no models available for behavior benchmark.")
                        ppl_by_model = None
                        if self._perplexity_results:
                            missing = [n for n in run_names if n not in self._perplexity_results]
                            if missing:
                                have = sorted(self._perplexity_results.keys())
                                logger.warning(
                                    f"{spec.id}: perplexity results missing for {missing}; "
                                    f"continuing behavior without ppl_by_model (have={have})."
                                )
                            else:
                                ppl_by_model = {
                                    n: float(self._perplexity_results[n].perplexity) for n in run_names
                                }
                        else:
                            logger.warning(
                                f"{spec.id}: no perplexity results available; "
                                "continuing behavior without ppl_by_model."
                            )
                        benchmark = BehaviorBenchmark(
                            suite_file=str(spec.config.suite_file),
                            tokenizer_config=spec.config.tokenizer,
                            device=self.device,
                        )
                        result = benchmark.run_multi_isolated(
                            model_names=run_names,
                            load_model=load_model,
                            unload_model=unload_model,
                            benchmark_id=str(spec.id),
                            output_dir=self.output_dir,
                            baseline_name=self.baseline_name,
                            seed=int(spec.config.seed),
                            max_new_tokens=int(spec.config.max_new_tokens),
                            context_window=spec.config.context_window,
                            dump_attention=bool(spec.config.dump_attention),
                            dump_attention_max_tokens=int(spec.config.dump_attention_max_tokens),
                            dump_attention_max_heads=int(spec.config.dump_attention_max_heads),
                            dump_attention_anchor=str(spec.config.dump_attention_anchor),
                            ppl_by_model=ppl_by_model,
                        )
                        self._behavior_runs[str(spec.id)] = result
                        self._behavior_results.update(_to_behavioral_results_dict(result))
                        self._save_incremental_artifacts("behavior")

                    case BenchmarkType.CONTEXT:
                        assert isinstance(spec.config, ContextBenchmarkConfig)
                        benchmark = BenchmarkContext(spec.config, self.device)
                        raw_dir = self.output_dir / "raw"
                        raw_path = raw_dir / f"{spec.id}_context_raw.png"
                        with with_plotter(
                            title=f"Context raw • {spec.id}",
                            enabled=bool(getattr(spec, "realtime", False)),
                        ) as plotter:
                            for name in models_to_run:
                                if name not in model_names:
                                    continue
                                m = _with_model(name)
                                try:
                                    result = benchmark.run(
                                        m,
                                        name,
                                        on_step=(
                                            (
                                                lambda r, mn=str(name): plotter.log(
                                                    **{
                                                        f"{mn}/ppl": float(r.sweep[-1].ppl)
                                                        if r.sweep
                                                        else float("nan"),
                                                        f"{mn}/decode_tok_s": float(
                                                            r.decode[-1].decode_tok_per_s
                                                        )
                                                        if r.decode
                                                        else float("nan"),
                                                    }
                                                )
                                            )
                                            if getattr(spec, "realtime", False)
                                            else None
                                        ),
                                    )
                                    self._context_results[name] = result
                                finally:
                                    unload_model(m)
                            if getattr(spec, "realtime", False):
                                try:
                                    plotter.save(raw_path)  # type: ignore[union-attr]
                                    raw_plot_paths.append(raw_path)
                                except Exception:
                                    pass
                        self._save_incremental_artifacts("context")

                    case _:
                        logger.warning(f"Skipping unsupported benchmark type: {spec.config.type}")

        paths = self._generate_final_artifacts()
        # Stitch all "raw" benchmark plots into one comparison image (if any).
        try:
            if raw_plot_paths:
                out_path = self.output_dir / "raw" / "benchmarks_raw_comparison.png"
                stitched = stitch_images(
                    images=raw_plot_paths,
                    out_path=out_path,
                    cols=2,
                    title="Raw benchmark plots (realtime)",
                )
                if stitched is not None:
                    paths["raw_benchmarks_comparison"] = stitched
        except Exception:
            pass
        return paths

    def _generate_final_artifacts(self) -> dict[str, Path]:
        """Generate the final multi-model artifacts from accumulated results."""
        logger.info("Generating final multi-model artifacts...")

        from benchmark.multi_model_artifacts import MultiModelArtifactGenerator

        generator = MultiModelArtifactGenerator(self.output_dir, self.baseline_name)
        all_behavioral = dict(self._behavior_results)

        paths = generator.generate_all(
            metadata=self.metadata,
            perplexity_results=self._perplexity_results,
            latency_results=self._latency_results,
            memory_results=self._memory_results,
            accuracy_results=self._accuracy_results,
            context_results=self._context_results,
            behavioral_results=(all_behavioral if all_behavioral else None),
            audit=self._audit or None,
            formats=self.suite.formats,
        )

        logger.success(f"Generated {len(paths)} artifacts in {self.output_dir}")
        for name, path in paths.items():
            logger.path(str(path), name)
        return paths

    def _run_accuracy_task_first_isolated(
        self,
        config: AccuracyBenchmarkConfig,
        models_to_run: list[str],
        load_model,
        unload_model,
        plotter=None,
    ) -> dict[str, AccuracyResult]:
        """Accuracy benchmark with per-model load/unload isolation."""
        from eval.logprob.completion.full_sequence import LogprobCompletionFullSequence
        from eval.logprob.completion.windowed import LogprobCompletionWindowed
        from eval.logprob.scorer import LogprobScorer
        from benchmark.accuracy.tasks.builder import BenchmarkAccuracyTaskBuilder

        run_names = [str(m) for m in models_to_run if str(m).strip()]
        results: dict[str, AccuracyResult] = {name: AccuracyResult(model_name=name) for name in run_names}

        benchmark = BenchmarkAccuracy(config, self.device)
        ctxw = int(config.context_window) if config.context_window is not None else None

        for task_name in list(config.tasks):
            t_name = str(task_name).strip().lower()
            if not t_name:
                continue
            logger.subheader(f"accuracy:{t_name}")

            # Build task once using the first model (then unload).
            first_name = next(iter(run_names), None)
            if first_name is None:
                break
            first_model = load_model(first_name)
            first_model.eval()
            try:
                completion = (
                    LogprobCompletionWindowed(model=first_model, device=self.device, context_window=int(ctxw))
                    if ctxw is not None
                    else LogprobCompletionFullSequence(model=first_model, device=self.device)
                )
                scorer = LogprobScorer(tokenizer=benchmark.tokenizer, completion=completion)
                builder = BenchmarkAccuracyTaskBuilder(
                    scorer=scorer,
                    config=config,
                    output_dir=self.output_dir,
                    device=self.device,
                )
                task = builder.build(t_name)
            finally:
                unload_model(first_model)

            # Now run for each model, loading/unloading each time.
            for model_name in run_names:
                model = load_model(model_name)
                model.eval()
                try:
                    completion = (
                        LogprobCompletionWindowed(model=model, device=self.device, context_window=int(ctxw))
                        if ctxw is not None
                        else LogprobCompletionFullSequence(model=model, device=self.device)
                    )
                    task.scorer = LogprobScorer(tokenizer=benchmark.tokenizer, completion=completion)
                    try:
                        if plotter is not None:
                            task.set_live_plotter(plotter, series=f"{model_name}:{t_name}")
                    except Exception:
                        pass
                    task_result = task.run(model=model, model_name=model_name)
                    results[model_name].tasks.append(task_result)
                    logger.metric(f"{model_name}:{t_name}", task_result.accuracy * 100.0, "%")
                finally:
                    unload_model(model)

        return results


def _to_behavioral_results_dict(result: BehaviorResult) -> dict[str, dict]:
    """Convert unified behavior result into the dict shape expected by MultiModelArtifactGenerator."""
    out: dict[str, dict] = {}
    models = list(result.summaries.keys())
    cats = sorted({c.category for c in result.cases})

    # Precompute per-category weighted accuracies per model.
    by_cat: dict[str, dict[str, float]] = {m: {} for m in models}
    for cat in cats:
        cat_cases = [cr for cr in result.results if cr.case.category == cat]
        if not cat_cases:
            raise RuntimeError(f"BehaviorResult missing category cases: {cat!r}")
        for m in models:
            ssum = float(sum(cr.outputs[m].final_score for cr in cat_cases))
            smax = float(sum(cr.outputs[m].difficulty_weight for cr in cat_cases))
            by_cat[m][cat] = float(ssum / smax) if smax > 0 else 0.0

    for m, s in result.summaries.items():
        out[str(m)] = {
            # Historical keys used by multi-model artifacts.
            "exact_match_rate": float(s.hard_accuracy),
            "partial_or_better_rate": float(s.soft_accuracy),
            "summary": {
                "hard_accuracy": float(s.hard_accuracy),
                "soft_accuracy": float(s.soft_accuracy),
                "weighted_accuracy": float(s.weighted_accuracy),
                "exact_count": int(s.exact),
                "contained_count": int(s.contained),
                "none_count": int(s.none),
                "total_tests": int(s.n),
            },
            "by_category": {
                cat: {
                    # MultiModelArtifactGenerator expects "soft_accuracy" here; we map to our weighted accuracy.
                    "soft_accuracy": float(by_cat[str(m)][cat]),
                    "weighted_accuracy": float(by_cat[str(m)][cat]),
                }
                for cat in cats
            },
        }
    return out

    def _run_accuracy_task_first(
        self,
        config: AccuracyBenchmarkConfig,
        models: dict[str, nn.Module],
        models_to_run: list[str],
        plotter=None,
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
                raise

            # Now run for each model, swapping the scorer
            for model_name in models_to_run:
                if model_name not in models:
                    raise RuntimeError(
                        f"Model '{model_name}' not found (requested by accuracy benchmark; task='{t_name}')."
                    )

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
                    if plotter is not None:
                        task.set_live_plotter(plotter, series=f"{model_name}:{t_name}")
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
