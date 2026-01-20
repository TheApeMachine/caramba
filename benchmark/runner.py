"""Benchmark runner: orchestrating all benchmarks and artifact generation.

The runner is the entry point for benchmarking. It executes all configured
benchmarks on teacher and student models, then generates paper-ready
artifacts (CSV, JSON, PNG, LaTeX).
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from benchmark.artifacts import ArtifactGenerator, ExperimentMetadata
from benchmark.accuracy import BenchmarkAccuracy, AccuracyResult
from benchmark.behavior import BehaviorBenchmark, BehaviorResult
from benchmark.behavioral_v2 import BenchmarkBehavioralV2, BehavioralV2Result
from benchmark.context import BenchmarkContext, ContextResult
from benchmark.latency import LatencyBenchmark, LatencyResult
from benchmark.memory import MemoryBenchmark, MemoryResult
from benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from config.benchmark import (
    AccuracyBenchmarkConfig,
    BenchmarkSuite,
    BenchmarkType,
    BehaviorBenchmarkConfig,
    BehavioralV2BenchmarkConfig,
    ContextBenchmarkConfig,
    LatencyBenchmarkConfig,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
)
from console import logger


class BenchmarkRunner:
    """Runs all configured benchmarks and generates artifacts.

    Orchestrates perplexity, latency, and memory benchmarks, comparing
    teacher and student models to quantify the upcycling trade-offs.
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        device: torch.device,
        metadata: ExperimentMetadata,
    ) -> None:
        """Set up the runner with benchmark suite and experiment metadata."""
        self.suite = suite
        self.device = device
        self.metadata = metadata
        self.output_dir = Path(suite.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        teacher: nn.Module | None,
        student: nn.Module | None,
    ) -> dict[str, Path]:
        """Run all benchmarks and generate artifacts.

        Returns a dict mapping artifact names to their file paths.
        """
        logger.header("Benchmarks", f"{len(self.suite.benchmarks)} configured")

        # NOTE: repeats are intended for statistical confidence; we aggregate
        # repeats rather than cherry-picking "best" runs.
        teacher_perplexity_runs: list[PerplexityResult] = []
        student_perplexity_runs: list[PerplexityResult] = []
        teacher_latency_runs: list[LatencyResult] = []
        student_latency_runs: list[LatencyResult] = []
        teacher_memory_runs: list[MemoryResult] = []
        student_memory_runs: list[MemoryResult] = []
        behavior: BehaviorResult | None = None
        behavioral_v2: BehavioralV2Result | None = None
        teacher_accuracy: AccuracyResult | None = None
        student_accuracy: AccuracyResult | None = None
        teacher_accuracy_runs: list[AccuracyResult] = []
        student_accuracy_runs: list[AccuracyResult] = []
        teacher_context_runs: list[ContextResult] = []
        student_context_runs: list[ContextResult] = []

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
            except Exception as e:
                logger.warning(f"Failed to move model to {device}: {e!r}")

        def _merge_perplexity(runs: list[PerplexityResult]) -> PerplexityResult | None:
            if not runs:
                return None
            model_name = runs[0].model_name
            total_tokens = int(sum(int(r.num_tokens) for r in runs))
            total_batches = int(sum(int(r.num_batches) for r in runs))
            if total_tokens <= 0:
                return PerplexityResult(
                    model_name=model_name,
                    perplexity=float("inf"),
                    loss=float("inf"),
                    num_tokens=0,
                    num_batches=total_batches,
                )
            total_loss = float(sum(float(r.loss) * float(r.num_tokens) for r in runs))
            avg_loss = total_loss / float(total_tokens)
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            return PerplexityResult(
                model_name=model_name,
                perplexity=float(ppl),
                loss=float(avg_loss),
                num_tokens=total_tokens,
                num_batches=total_batches,
            )

        def _merge_latency(runs: list[LatencyResult]) -> LatencyResult | None:
            if not runs:
                return None
            out = LatencyResult(model_name=runs[0].model_name)
            for r in runs:
                out.measurements.extend(r.measurements)
            return out

        def _merge_memory(runs: list[MemoryResult]) -> MemoryResult | None:
            if not runs:
                return None
            out = MemoryResult(model_name=runs[0].model_name)
            out.kvcache_analysis = runs[0].kvcache_analysis

            # Ensure KV-cache analysis is consistent across repeats.
            if out.kvcache_analysis is not None:
                ref = asdict(out.kvcache_analysis)
                for r in runs[1:]:
                    if r.kvcache_analysis is None:
                        raise ValueError(
                            "Memory benchmark repeats produced inconsistent kvcache_analysis "
                            f"(expected present for {out.model_name})."
                        )
                    if asdict(r.kvcache_analysis) != ref:
                        raise ValueError(
                            "Memory benchmark repeats produced inconsistent kvcache_analysis "
                            f"for {out.model_name}."
                        )

            for r in runs:
                out.measurements.extend(r.measurements)
            return out

        def _merge_accuracy(runs: list[AccuracyResult]) -> AccuracyResult | None:
            if not runs:
                return None
            model_name = runs[0].model_name
            # Aggregate by (task, split) summing correct/total; recompute accuracy.
            agg: dict[tuple[str, str], dict[str, float]] = {}
            for r in runs:
                for t in r.tasks:
                    key = (str(t.task), str(t.split))
                    item = agg.setdefault(
                        key,
                        {"correct": 0.0, "total": 0.0, "elapsed_sum": 0.0, "n": 0.0},
                    )
                    item["correct"] += float(int(t.correct))
                    item["total"] += float(int(t.total))
                    item["elapsed_sum"] += float(getattr(t, "elapsed_seconds", 0.0))
                    item["n"] += 1.0

            out = AccuracyResult(model_name=str(model_name))
            from collector.measurement.accuracy.task import TaskAccuracy

            for (task, split), v in sorted(agg.items()):
                tot = int(v["total"])
                cor = int(v["correct"])
                acc = float(cor) / float(tot) if tot > 0 else 0.0
                elapsed = float(v["elapsed_sum"] / max(1.0, v["n"]))
                out.tasks.append(
                    TaskAccuracy(
                        task=str(task),
                        split=str(split),
                        accuracy=float(acc),
                        correct=int(cor),
                        total=int(tot),
                        samples=[],
                        elapsed_seconds=float(elapsed),
                    )
                )
            return out

        def _merge_context(runs: list[ContextResult]) -> ContextResult | None:
            if not runs:
                return None
            out = ContextResult(model_name=str(runs[0].model_name))
            for r in runs:
                out.sweep.extend(r.sweep)
                out.decode.extend(r.decode)
            return out

        for spec in self.suite.benchmarks:
            logger.subheader(f"{spec.id} ({spec.config.type})")

            for _ in range(spec.repeats):
                match spec.config.type:
                    case BenchmarkType.PERPLEXITY:
                        assert isinstance(spec.config, PerplexityBenchmarkConfig)
                        benchmark = PerplexityBenchmark(spec.config, self.device)

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                result = benchmark.run(teacher, "teacher")
                                teacher_perplexity_runs.append(result)
                                logger.metric("teacher", result.perplexity, " ppl")

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                result = benchmark.run(student, "student")
                                student_perplexity_runs.append(result)
                                logger.metric("student", result.perplexity, " ppl")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)
                        # Latency is resource-sensitive: isolate the active model on the accelerator.
                        cpu = torch.device("cpu")

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                # Move student away (if present) while benchmarking teacher.
                                if student is not None and student is not teacher:
                                    _try_move_model(student, cpu)
                                    _gc_and_empty_cache()
                                _try_move_model(teacher, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(teacher, "teacher")
                                teacher_latency_runs.append(result)
                                logger.metric(
                                    "teacher", result.avg_tokens_per_second, " tok/s"
                                )

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                # Move teacher away (if present) while benchmarking student.
                                if teacher is not None and teacher is not student:
                                    _try_move_model(teacher, cpu)
                                    _gc_and_empty_cache()
                                _try_move_model(student, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(student, "student")
                                student_latency_runs.append(result)
                                logger.metric(
                                    "student", result.avg_tokens_per_second, " tok/s"
                                )
                        # Restore both models to the benchmark device for subsequent benchmarks.
                        _try_move_model(teacher, self.device)
                        _try_move_model(student, self.device)
                        _gc_and_empty_cache()

                    case BenchmarkType.MEMORY:
                        assert isinstance(spec.config, MemoryBenchmarkConfig)
                        benchmark = MemoryBenchmark(spec.config, self.device)
                        # Memory is extremely resource-sensitive: isolate the active model on the accelerator.
                        cpu = torch.device("cpu")

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                if student is not None and student is not teacher:
                                    _try_move_model(student, cpu)
                                    _gc_and_empty_cache()
                                _try_move_model(teacher, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(teacher, "teacher")
                                teacher_memory_runs.append(result)
                                if result.kvcache_analysis:
                                    logger.metric(
                                        "teacher",
                                        result.kvcache_analysis.bytes_per_token_fp16,
                                        " bytes/tok",
                                    )

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                if teacher is not None and teacher is not student:
                                    _try_move_model(teacher, cpu)
                                    _gc_and_empty_cache()
                                _try_move_model(student, self.device)
                                _gc_and_empty_cache()
                                result = benchmark.run(student, "student")
                                student_memory_runs.append(result)
                                if result.kvcache_analysis:
                                    kv_bytes = (
                                        result.kvcache_analysis.bytes_per_token_dba_fp16
                                        or result.kvcache_analysis.bytes_per_token_fp16
                                    )
                                    logger.metric("student", kv_bytes, " bytes/tok")
                        # Restore both models to the benchmark device for subsequent benchmarks.
                        _try_move_model(teacher, self.device)
                        _try_move_model(student, self.device)
                        _gc_and_empty_cache()

                    case BenchmarkType.ACCURACY:
                        assert isinstance(spec.config, AccuracyBenchmarkConfig)
                        benchmark = BenchmarkAccuracy(spec.config, self.device)

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                r = benchmark.run(teacher, "teacher", output_dir=self.output_dir)
                                teacher_accuracy_runs.append(r)
                                logger.metric("teacher", r.micro_accuracy * 100.0, "% micro-acc")

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                r = benchmark.run(student, "student", output_dir=self.output_dir)
                                student_accuracy_runs.append(r)
                                logger.metric("student", r.micro_accuracy * 100.0, "% micro-acc")

                    case BenchmarkType.BEHAVIOR:
                        assert isinstance(spec.config, BehaviorBenchmarkConfig)
                        if teacher is None or student is None:
                            logger.warning("Behavior benchmark requires both teacher and student (skipping).")
                        else:
                            benchmark = BehaviorBenchmark(spec.config, self.device)
                            result = benchmark.run(
                                teacher=teacher,
                                student=student,
                                benchmark_id=str(spec.id),
                                output_dir=self.output_dir,
                            )
                            behavior = result
                            logger.metric(
                                "teacher", float(result.teacher_accuracy) * 100.0, "% acc"
                            )
                            logger.metric(
                                "student", float(result.student_accuracy) * 100.0, "% acc"
                            )

                    case BenchmarkType.BEHAVIORAL_V2:
                        assert isinstance(spec.config, BehavioralV2BenchmarkConfig)
                        if teacher is None or student is None:
                            logger.warning("Behavioral v2 benchmark requires both teacher and student (skipping).")
                        else:
                            benchmark = BenchmarkBehavioralV2(spec.config, self.device)
                            result = benchmark.run(
                                teacher=teacher,
                                student=student,
                                output_dir=self.output_dir,
                            )
                            behavioral_v2 = result
                            logger.metric(
                                "teacher", result.teacher_exact_rate * 100.0, "% exact"
                            )
                            logger.metric(
                                "student", result.student_exact_rate * 100.0, "% exact"
                            )

                    case BenchmarkType.CONTEXT:
                        assert isinstance(spec.config, ContextBenchmarkConfig)
                        benchmark = BenchmarkContext(spec.config, self.device)
                        # Context sweep is memory-sensitive: avoid keeping a second 1B model
                        # resident on the accelerator. Best-effort isolate by moving the
                        # non-active model to CPU and clearing backend caches.
                        cpu = torch.device("cpu")

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                # Move student away (if present) while sweeping teacher.
                                if student is not None and student is not teacher:
                                    _try_move_model(student, cpu)
                                    _gc_and_empty_cache()
                                teacher_context = benchmark.run(teacher, "teacher")
                                teacher_context_runs.append(teacher_context)
                                # Report last decode-tps at max context.
                                try:
                                    decode = teacher_context.decode
                                    xs = [
                                        m.decode_tok_per_s
                                        for m in decode
                                        if m.ok and m.decode_tok_per_s == m.decode_tok_per_s
                                    ]
                                except Exception as e:
                                    logger.error(
                                        "Failed to collect teacher decode rates from teacher_context.decode "
                                        f"(reading m.decode_tok_per_s): {e!r}"
                                    )
                                else:
                                    if xs:
                                        try:
                                            logger.metric("teacher", float(xs[-1]), " tok/s@ctx")
                                        except Exception as e:
                                            logger.error(
                                                "Failed to emit teacher tok/s@ctx via logger.metric "
                                                f"(from teacher_context.decode / m.decode_tok_per_s): {e!r}"
                                            )
                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                # Move teacher away (if present) while sweeping student.
                                if teacher is not None and teacher is not student:
                                    _try_move_model(teacher, cpu)
                                    _gc_and_empty_cache()
                                # Ensure student is on the benchmark device (if we moved it earlier).
                                _try_move_model(student, self.device)
                                _gc_and_empty_cache()
                                student_context = benchmark.run(student, "student")
                                student_context_runs.append(student_context)
                                try:
                                    decode = student_context.decode
                                    xs = [
                                        m.decode_tok_per_s
                                        for m in decode
                                        if m.ok and m.decode_tok_per_s == m.decode_tok_per_s
                                    ]
                                except Exception as e:
                                    logger.error(
                                        "Failed to collect student decode rates from student_context.decode "
                                        f"(reading m.decode_tok_per_s): {e!r}"
                                    )
                                else:
                                    if xs:
                                        try:
                                            logger.metric("student", float(xs[-1]), " tok/s@ctx")
                                        except Exception as e:
                                            logger.error(
                                                "Failed to emit student tok/s@ctx via logger.metric "
                                                f"(from student_context.decode / m.decode_tok_per_s): {e!r}"
                                            )

                        # Restore both models to the benchmark device for subsequent benchmarks.
                        # (Best-effort; does not fail the run if a move isn't supported.)
                        _try_move_model(teacher, self.device)
                        _try_move_model(student, self.device)
                        _gc_and_empty_cache()

                    case _:
                        logger.warning(
                            f"skipping unsupported benchmark type: {spec.config.type}"
                        )

        teacher_perplexity = _merge_perplexity(teacher_perplexity_runs)
        student_perplexity = _merge_perplexity(student_perplexity_runs)
        teacher_latency = _merge_latency(teacher_latency_runs)
        student_latency = _merge_latency(student_latency_runs)
        teacher_memory = _merge_memory(teacher_memory_runs)
        student_memory = _merge_memory(student_memory_runs)
        teacher_accuracy = _merge_accuracy(teacher_accuracy_runs)
        student_accuracy = _merge_accuracy(student_accuracy_runs)
        teacher_context = _merge_context(teacher_context_runs)
        student_context = _merge_context(student_context_runs)

        # Generate artifacts
        logger.info("Generating artifacts...")
        generator = ArtifactGenerator(self.output_dir)
        paths = generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=teacher_perplexity,
            student_perplexity=student_perplexity,
            teacher_latency=teacher_latency,
            student_latency=student_latency,
            teacher_memory=teacher_memory,
            student_memory=student_memory,
            teacher_accuracy=teacher_accuracy,
            student_accuracy=student_accuracy,
            behavior=behavior,
            behavioral_v2=behavioral_v2,
            teacher_context=teacher_context,
            student_context=student_context,
            formats=self.suite.formats,
        )

        logger.success(f"Generated {len(paths)} artifacts in {self.output_dir}")
        for name, path in paths.items():
            logger.path(str(path), name)

        return paths


class QuickBenchmark:
    """Quick benchmark for sanity checking during development.

    Runs a minimal set of benchmarks with reduced iterations,
    useful for verifying a model works before full benchmarking.
    """

    def __init__(self, device: torch.device) -> None:
        """Set up the quick benchmark."""
        self.device = device

    def run(
        self,
        model: nn.Module,
        dataset_path: str,
        num_batches: int = 10,
    ) -> dict[str, float]:
        """Run quick benchmark and return key metrics."""
        model.eval()

        ppl_config = PerplexityBenchmarkConfig(
            dataset=dataset_path,
            block_size=512,
            batch_size=1,
            num_batches=num_batches,
        )
        ppl_benchmark = PerplexityBenchmark(ppl_config, self.device)
        ppl_result = ppl_benchmark.run(model, "model")

        mem_config = MemoryBenchmarkConfig(
            sequence_lengths=[512],
            batch_sizes=[1],
            quantization_modes=["fp16"],
        )
        mem_benchmark = MemoryBenchmark(mem_config, self.device)
        mem_result = mem_benchmark.run(model, "model")

        kv_bytes = 0.0
        if mem_result.kvcache_analysis:
            if mem_result.kvcache_analysis.bytes_per_token_dba_fp16:
                kv_bytes = mem_result.kvcache_analysis.bytes_per_token_dba_fp16
            else:
                kv_bytes = mem_result.kvcache_analysis.bytes_per_token_fp16

        return {
            "perplexity": ppl_result.perplexity,
            "loss": ppl_result.loss,
            "kv_bytes_per_token": kv_bytes,
        }
