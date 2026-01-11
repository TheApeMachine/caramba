"""Benchmark runner: orchestrating all benchmarks and artifact generation.

The runner is the entry point for benchmarking. It executes all configured
benchmarks on teacher and student models, then generates paper-ready
artifacts (CSV, JSON, PNG, LaTeX).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from caramba.benchmark.artifacts import ArtifactGenerator, ExperimentMetadata
from caramba.benchmark.accuracy import BenchmarkAccuracy, AccuracyResult
from caramba.benchmark.behavior import BehaviorBenchmark, BehaviorResult
from caramba.benchmark.context import BenchmarkContext, ContextResult
from caramba.benchmark.latency import LatencyBenchmark, LatencyResult
from caramba.benchmark.memory import MemoryBenchmark, MemoryResult
from caramba.benchmark.perplexity import PerplexityBenchmark, PerplexityResult
from caramba.config.benchmark import (
    AccuracyBenchmarkConfig,
    BenchmarkSuite,
    BenchmarkType,
    BehaviorBenchmarkConfig,
    ContextBenchmarkConfig,
    LatencyBenchmarkConfig,
    MemoryBenchmarkConfig,
    PerplexityBenchmarkConfig,
)
from caramba.console import logger


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

        teacher_perplexity: PerplexityResult | None = None
        student_perplexity: PerplexityResult | None = None
        teacher_latency: LatencyResult | None = None
        student_latency: LatencyResult | None = None
        teacher_memory: MemoryResult | None = None
        student_memory: MemoryResult | None = None
        behavior: BehaviorResult | None = None
        teacher_accuracy: AccuracyResult | None = None
        student_accuracy: AccuracyResult | None = None
        teacher_context: ContextResult | None = None
        student_context: ContextResult | None = None

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
                                if (
                                    teacher_perplexity is None
                                    or result.perplexity < teacher_perplexity.perplexity
                                ):
                                    teacher_perplexity = result
                                logger.metric("teacher", result.perplexity, " ppl")

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                result = benchmark.run(student, "student")
                                if (
                                    student_perplexity is None
                                    or result.perplexity < student_perplexity.perplexity
                                ):
                                    student_perplexity = result
                                logger.metric("student", result.perplexity, " ppl")

                    case BenchmarkType.LATENCY:
                        assert isinstance(spec.config, LatencyBenchmarkConfig)
                        benchmark = LatencyBenchmark(spec.config, self.device)

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                result = benchmark.run(teacher, "teacher")
                                if teacher_latency is None:
                                    teacher_latency = result
                                logger.metric(
                                    "teacher", result.avg_tokens_per_second, " tok/s"
                                )

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                result = benchmark.run(student, "student")
                                if student_latency is None:
                                    student_latency = result
                                logger.metric(
                                    "student", result.avg_tokens_per_second, " tok/s"
                                )

                    case BenchmarkType.MEMORY:
                        assert isinstance(spec.config, MemoryBenchmarkConfig)
                        benchmark = MemoryBenchmark(spec.config, self.device)

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                result = benchmark.run(teacher, "teacher")
                                if teacher_memory is None:
                                    teacher_memory = result
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
                                result = benchmark.run(student, "student")
                                if student_memory is None:
                                    student_memory = result
                                if result.kvcache_analysis:
                                    kv_bytes = (
                                        result.kvcache_analysis.bytes_per_token_dba_fp16
                                        or result.kvcache_analysis.bytes_per_token_fp16
                                    )
                                    logger.metric("student", kv_bytes, " bytes/tok")

                    case BenchmarkType.ACCURACY:
                        assert isinstance(spec.config, AccuracyBenchmarkConfig)
                        benchmark = BenchmarkAccuracy(spec.config, self.device)

                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                teacher_accuracy = benchmark.run(teacher, "teacher", output_dir=self.output_dir)
                                logger.metric("teacher", teacher_accuracy.micro_accuracy * 100.0, "% micro-acc")

                        if "student" in spec.models:
                            if student is None:
                                logger.warning("Benchmark requested student model, but none is available (skipping).")
                            else:
                                student_accuracy = benchmark.run(student, "student", output_dir=self.output_dir)
                                logger.metric("student", student_accuracy.micro_accuracy * 100.0, "% micro-acc")

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

                    case BenchmarkType.CONTEXT:
                        assert isinstance(spec.config, ContextBenchmarkConfig)
                        benchmark = BenchmarkContext(spec.config, self.device)
                        if "teacher" in spec.models:
                            if teacher is None:
                                logger.warning("Benchmark requested teacher model, but none is available (skipping).")
                            else:
                                teacher_context = benchmark.run(teacher, "teacher")
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
                                student_context = benchmark.run(student, "student")
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

                    case _:
                        logger.warning(
                            f"skipping unsupported benchmark type: {spec.config.type}"
                        )

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
