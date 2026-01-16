"""Behavioral V2 benchmark: template-based behavioral evaluation with downstream tasks.

This benchmark integrates the v2 behavioral test suite into the manifest system,
providing:
- Template-based test generation (500+ tests across 18 categories)
- Teacher vs student comparison with match quality tracking (NONE/PARTIAL/EXACT)
- Optional downstream HF tasks (winogrande, arc_easy, etc.)
- Unified results with the new scoring system
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from caramba.config.benchmark import BehavioralV2BenchmarkConfig
from caramba.console import logger
from caramba.data.tokenizers.builder import TokenizerBuilder

# Import the v2 behavioral suite
from research.dba.behavioral_suite_v2 import generate_suite, GeneratedSuite
from research.dba.behavioral_suite_v2.scoring import (
    BehavioralScorer,
    MatchQuality,
    classify_match,
)


@dataclass
class BehavioralV2Result:
    """Results from behavioral v2 evaluation."""

    # Per-model summaries
    teacher_summary: dict[str, Any] = field(default_factory=dict)
    student_summary: dict[str, Any] = field(default_factory=dict)

    # Head-to-head comparison
    comparison: dict[str, Any] = field(default_factory=dict)

    # Per-category breakdowns
    teacher_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    student_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Downstream task results (if any)
    downstream_results: dict[str, Any] = field(default_factory=dict)

    # Raw data for detailed analysis
    scorer: BehavioralScorer | None = None

    @property
    def teacher_exact_rate(self) -> float:
        """Teacher exact match rate."""
        return self.teacher_summary.get("exact_match_rate", 0.0)

    @property
    def student_exact_rate(self) -> float:
        """Student exact match rate."""
        return self.student_summary.get("exact_match_rate", 0.0)

    @property
    def teacher_partial_or_better_rate(self) -> float:
        """Teacher partial or exact match rate."""
        return self.teacher_summary.get("partial_or_better_rate", 0.0)

    @property
    def student_partial_or_better_rate(self) -> float:
        """Student partial or exact match rate."""
        return self.student_summary.get("partial_or_better_rate", 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excluding scorer)."""
        return {
            "teacher_summary": self.teacher_summary,
            "student_summary": self.student_summary,
            "comparison": self.comparison,
            "teacher_by_category": self.teacher_by_category,
            "student_by_category": self.student_by_category,
            "downstream_results": self.downstream_results,
        }


class BenchmarkBehavioralV2:
    """Run behavioral v2 evaluation comparing teacher and student models."""

    def __init__(
        self,
        config: BehavioralV2BenchmarkConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.tokenizer = TokenizerBuilder().build(config.tokenizer)

    def run(
        self,
        teacher: nn.Module,
        student: nn.Module,
        output_dir: Path | None = None,
    ) -> BehavioralV2Result:
        """Run behavioral evaluation on both models."""
        teacher.eval()
        student.eval()

        # Generate test suite
        logger.info(f"Generating test suite (seed={self.config.seed})...")
        suite = generate_suite(
            seed=self.config.seed,
            tests_per_category=self.config.tests_per_category,
            category_counts=self.config.category_counts,
        )

        # Filter categories if specified
        tests = suite.tests
        if self.config.categories:
            tests = [t for t in tests if t.category in self.config.categories]
            logger.info(f"Filtered to {len(tests)} tests in categories: {self.config.categories}")

        logger.info(f"Running {len(tests)} behavioral tests...")

        # Create scorer
        scorer = BehavioralScorer()

        # Register all tests
        for test in tests:
            scorer.add_test(test.id, expected=test.expected, prompt=test.prompt)

        # Run inference on both models
        count = 0
        for test in tests:
            count += 1
            if self.config.stream_live and count % self.config.stream_every == 0:
                logger.info(f"  [{count}/{len(tests)}] {test.id}")

            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(test.prompt)

            # Get teacher output
            teacher_output = self._generate(teacher, prompt_tokens)
            scorer.add_output(test.id, "teacher", teacher_output)

            # Get student output
            student_output = self._generate(student, prompt_tokens)
            scorer.add_output(test.id, "student", student_output)

        # Get summaries
        teacher_summary = scorer.get_model_summary("teacher")
        student_summary = scorer.get_model_summary("student")
        comparison = scorer.compare_models("teacher", "student")

        # Log key metrics
        logger.metric("teacher_exact", teacher_summary["exact_match_rate"] * 100, "%")
        logger.metric("student_exact", student_summary["exact_match_rate"] * 100, "%")
        logger.metric("teacher_partial+", teacher_summary["partial_or_better_rate"] * 100, "%")
        logger.metric("student_partial+", student_summary["partial_or_better_rate"] * 100, "%")

        # Per-category breakdown
        teacher_by_cat = self._compute_category_breakdown(scorer, "teacher", tests)
        student_by_cat = self._compute_category_breakdown(scorer, "student", tests)

        # Run downstream tasks if configured
        downstream_results = {}
        if self.config.downstream_tasks:
            downstream_results = self._run_downstream_tasks(teacher, student)

        result = BehavioralV2Result(
            teacher_summary=teacher_summary,
            student_summary=student_summary,
            comparison=comparison,
            teacher_by_category=teacher_by_cat,
            student_by_category=student_by_cat,
            downstream_results=downstream_results,
            scorer=scorer,
        )

        # Save results if output dir provided
        if output_dir:
            self._save_results(result, scorer, output_dir)

        return result

    def _generate(self, model: nn.Module, prompt_tokens: list[int]) -> str:
        """Generate output from a model given prompt tokens."""
        # Convert to tensor
        input_ids = torch.tensor([prompt_tokens], device=self.device)

        # Generate
        with torch.no_grad():
            # Simple greedy generation
            for _ in range(self.config.max_new_tokens):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Get last token logits
                next_logits = logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)

                # Check for EOS (assume token 0 or handle gracefully)
                if next_token.item() == 0:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Apply context window limit if specified
                if self.config.context_window and input_ids.shape[1] > self.config.context_window:
                    break

        # Decode only the generated part
        generated_ids = input_ids[0, len(prompt_tokens):].tolist()
        return self.tokenizer.decode(generated_ids)

    def _compute_category_breakdown(
        self,
        scorer: BehavioralScorer,
        model_id: str,
        tests: list,
    ) -> dict[str, dict[str, Any]]:
        """Compute per-category statistics."""
        from collections import defaultdict

        by_category: dict[str, list] = defaultdict(list)
        for test in tests:
            if test.id in scorer.tests and model_id in scorer.tests[test.id].results:
                result = scorer.tests[test.id].results[model_id]
                by_category[test.category].append(result)

        breakdown = {}
        for category, results in by_category.items():
            n = len(results)
            exact = sum(1 for r in results if r.quality == MatchQuality.EXACT)
            partial = sum(1 for r in results if r.quality == MatchQuality.PARTIAL)
            breakdown[category] = {
                "total": n,
                "exact": exact,
                "partial": partial,
                "none": n - exact - partial,
                "exact_rate": exact / n if n > 0 else 0,
                "partial_or_better_rate": (exact + partial) / n if n > 0 else 0,
            }

        return breakdown

    def _run_downstream_tasks(
        self,
        teacher: nn.Module,
        student: nn.Module,
    ) -> dict[str, Any]:
        """Run downstream HF tasks (winogrande, arc_easy, etc.)."""
        from caramba.config.benchmark import AccuracyBenchmarkConfig
        from caramba.benchmark.accuracy import BenchmarkAccuracy

        results = {}

        # Create accuracy config for downstream tasks
        acc_config = AccuracyBenchmarkConfig(
            tasks=self.config.downstream_tasks,
            tokenizer=self.config.tokenizer,
            limit=self.config.downstream_limit,
            context_window=self.config.context_window,
            stream_live=self.config.stream_live,
            stream_every=self.config.stream_every,
        )

        benchmark = BenchmarkAccuracy(acc_config, self.device)

        logger.info(f"Running downstream tasks: {self.config.downstream_tasks}")

        teacher_result = benchmark.run(teacher, "teacher")
        student_result = benchmark.run(student, "student")

        results["teacher"] = {
            "micro_accuracy": teacher_result.micro_accuracy,
            "tasks": {t.task_name: t.accuracy for t in teacher_result.tasks},
        }
        results["student"] = {
            "micro_accuracy": student_result.micro_accuracy,
            "tasks": {t.task_name: t.accuracy for t in student_result.tasks},
        }

        logger.metric("teacher_downstream", teacher_result.micro_accuracy * 100, "%")
        logger.metric("student_downstream", student_result.micro_accuracy * 100, "%")

        return results

    def _save_results(
        self,
        result: BehavioralV2Result,
        scorer: BehavioralScorer,
        output_dir: Path,
    ) -> None:
        """Save results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "behavioral_v2_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.path(str(summary_path), "behavioral_v2_summary")

        # Save detailed results
        detailed_path = output_dir / "behavioral_v2_detailed.json"
        with open(detailed_path, "w") as f:
            json.dump(scorer.to_dict(), f, indent=2)
        logger.path(str(detailed_path), "behavioral_v2_detailed")

        # Save interesting cases
        interesting = scorer.get_interesting_cases("teacher", "student")
        interesting_path = output_dir / "behavioral_v2_interesting.json"
        with open(interesting_path, "w") as f:
            json.dump(interesting, f, indent=2)
        logger.path(str(interesting_path), "behavioral_v2_interesting")

        # Write log file if configured
        if self.config.log_file:
            log_path = output_dir / self.config.log_file
            with open(log_path, "w") as f:
                f.write("Behavioral V2 Evaluation Log\n")
                f.write("=" * 80 + "\n\n")

                f.write("Teacher Summary:\n")
                for k, v in result.teacher_summary.items():
                    f.write(f"  {k}: {v}\n")

                f.write("\nStudent Summary:\n")
                for k, v in result.student_summary.items():
                    f.write(f"  {k}: {v}\n")

                f.write("\nComparison:\n")
                for k, v in result.comparison.items():
                    f.write(f"  {k}: {v}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("Per-Test Results\n")
                f.write("=" * 80 + "\n\n")

                for test_id, test_result in scorer.tests.items():
                    f.write(f"\n--- {test_id} ---\n")
                    f.write(f"Expected: {test_result.expected}\n")
                    for model_id, match_result in test_result.results.items():
                        f.write(f"{model_id}: {match_result.quality.name}\n")
                        f.write(f"  First line: {match_result.first_line[:100]}\n")

            logger.path(str(log_path), "behavioral_v2_log")
