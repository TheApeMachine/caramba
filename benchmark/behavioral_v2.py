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
from research.dba.behavioral_suite_v2.weighted_scoring import (
    WeightedScorer,
    WeightedModelSummary,
    MatchType,
    classify_match_type,
    MATCH_SCORES,
    DIFFICULTY_WEIGHTS,
)
from research.dba.behavioral_suite_v2.weighted_visualizer import (
    generate_all_weighted_visualizations,
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


@dataclass
class BehavioralV2MultiResult:
    """Results from multi-model behavioral v2 evaluation."""

    # Per-model summaries: {model_name: summary_dict}
    model_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-model category breakdowns: {model_name: {category: stats}}
    model_by_category: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    # Pairwise comparisons: list of {model_a, model_b, wins_a, wins_b, ties}
    pairwise_comparisons: list[dict[str, Any]] = field(default_factory=list)

    # Downstream task results (if any): {model_name: results}
    downstream_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Raw scorer for detailed analysis
    scorer: BehavioralScorer | None = None

    # Baseline model name for delta calculations
    baseline_name: str | None = None

    # Weighted scoring summaries: {model_name: WeightedModelSummary}
    weighted_summaries: dict[str, WeightedModelSummary] = field(default_factory=dict)

    # Weighted scorer for detailed analysis
    weighted_scorer: WeightedScorer | None = None

    def get_exact_rate(self, model_name: str) -> float:
        """Get exact match rate for a model."""
        return self.model_summaries.get(model_name, {}).get("exact_match_rate", 0.0)

    def get_partial_or_better_rate(self, model_name: str) -> float:
        """Get partial or better match rate for a model."""
        return self.model_summaries.get(model_name, {}).get("partial_or_better_rate", 0.0)

    def get_hard_accuracy(self, model_name: str) -> float:
        """Get hard accuracy (EXACT only) for a model from weighted scoring."""
        if model_name in self.weighted_summaries:
            return self.weighted_summaries[model_name].hard_accuracy
        return 0.0

    def get_soft_accuracy(self, model_name: str) -> float:
        """Get soft accuracy (EXACT + CONTAINED) for a model from weighted scoring."""
        if model_name in self.weighted_summaries:
            return self.weighted_summaries[model_name].soft_accuracy
        return 0.0

    def get_weighted_accuracy(self, model_name: str) -> float:
        """Get weighted accuracy (difficulty-adjusted) for a model."""
        if model_name in self.weighted_summaries:
            return self.weighted_summaries[model_name].weighted_accuracy
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excluding scorer)."""
        return {
            "model_summaries": self.model_summaries,
            "model_by_category": self.model_by_category,
            "pairwise_comparisons": self.pairwise_comparisons,
            "downstream_results": self.downstream_results,
            "baseline_name": self.baseline_name,
            "weighted_summaries": {
                name: summary.to_dict()
                for name, summary in self.weighted_summaries.items()
            },
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

    def run_multi(
        self,
        models: dict[str, nn.Module],
        output_dir: Path | None = None,
        baseline_name: str | None = None,
    ) -> BehavioralV2MultiResult:
        """Run behavioral evaluation on N models.

        Args:
            models: Dict mapping model names to nn.Module instances
            output_dir: Optional directory to save results
            baseline_name: Name of baseline model for comparisons (default: first model)

        Returns:
            BehavioralV2MultiResult with per-model summaries and pairwise comparisons
        """
        for model in models.values():
            model.eval()

        model_names = list(models.keys())
        if baseline_name is None and model_names:
            baseline_name = model_names[0]

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

        logger.info(f"Running {len(tests)} behavioral tests on {len(models)} models...")

        # Create scorer
        scorer = BehavioralScorer()

        # Register all tests
        for test in tests:
            scorer.add_test(test.id, expected=test.expected, prompt=test.prompt)

        # Run inference on all models
        count = 0
        for test in tests:
            count += 1
            if self.config.stream_live and count % self.config.stream_every == 0:
                logger.info(f"  [{count}/{len(tests)}] {test.id}")

            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(test.prompt)

            # Get output from each model
            for model_name, model in models.items():
                output = self._generate(model, prompt_tokens)
                scorer.add_output(test.id, model_name, output)

        # Get per-model summaries
        model_summaries: dict[str, dict[str, Any]] = {}
        model_by_category: dict[str, dict[str, dict[str, Any]]] = {}

        for model_name in model_names:
            summary = scorer.get_model_summary(model_name)
            model_summaries[model_name] = summary
            model_by_category[model_name] = self._compute_category_breakdown(scorer, model_name, tests)

            # Log metrics
            logger.metric(
                f"{model_name}_exact",
                summary["exact_match_rate"] * 100,
                "%"
            )

        # Pairwise comparisons
        pairwise_comparisons: list[dict[str, Any]] = []
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                comparison = scorer.compare_models(model_a, model_b)
                pairwise_comparisons.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    **comparison,
                })

        # Compute weighted scores (hard/soft with difficulty weighting)
        weighted_scorer = WeightedScorer(baseline_name=baseline_name)
        weighted_summaries: dict[str, WeightedModelSummary] = {}

        # Register all tests with the weighted scorer
        for test in tests:
            weighted_scorer.add_test(test.id, expected=test.expected, category=test.category)

        # Add outputs from the existing scorer
        for test_id, test_result in scorer.tests.items():
            for model_id, match_result in test_result.results.items():
                weighted_scorer.add_output(test_id, model_id, match_result.actual)

        # Get weighted summaries for all models (including baseline for reference)
        for model_name in model_names:
            weighted_summaries[model_name] = weighted_scorer.get_model_summary(model_name)

            # Log weighted metrics
            ws = weighted_summaries[model_name]
            logger.metric(f"{model_name}_hard", ws.hard_accuracy * 100, "%")
            logger.metric(f"{model_name}_soft", ws.soft_accuracy * 100, "%")
            logger.metric(f"{model_name}_weighted", ws.weighted_accuracy * 100, "%")

        # Log baseline difficulty distribution
        baseline_summary = weighted_scorer.get_baseline_summary()
        if baseline_summary.get("total_tests", 0) > 0:
            dist = baseline_summary.get("difficulty_distribution", {})
            logger.info(
                f"Difficulty distribution (by {baseline_name}): "
                f"easy={dist.get('easy', 0)} medium={dist.get('medium', 0)} hard={dist.get('hard', 0)}"
            )

        # Run downstream tasks if configured
        downstream_results: dict[str, dict[str, Any]] = {}
        if self.config.downstream_tasks:
            downstream_results = self._run_downstream_tasks_multi(models)

        result = BehavioralV2MultiResult(
            model_summaries=model_summaries,
            model_by_category=model_by_category,
            pairwise_comparisons=pairwise_comparisons,
            downstream_results=downstream_results,
            scorer=scorer,
            baseline_name=baseline_name,
            weighted_summaries=weighted_summaries,
            weighted_scorer=weighted_scorer,
        )

        # Save results if output dir provided
        if output_dir:
            self._save_multi_results(result, scorer, output_dir, weighted_scorer)

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
            "tasks": {t.task: t.accuracy for t in teacher_result.tasks},
        }
        results["student"] = {
            "micro_accuracy": student_result.micro_accuracy,
            "tasks": {t.task: t.accuracy for t in student_result.tasks},
        }

        logger.metric("teacher_downstream", teacher_result.micro_accuracy * 100, "%")
        logger.metric("student_downstream", student_result.micro_accuracy * 100, "%")

        return results

    def _run_downstream_tasks_multi(
        self,
        models: dict[str, nn.Module],
    ) -> dict[str, dict[str, Any]]:
        """Run downstream HF tasks on N models."""
        from caramba.config.benchmark import AccuracyBenchmarkConfig
        from caramba.benchmark.accuracy import BenchmarkAccuracy

        results: dict[str, dict[str, Any]] = {}

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

        for model_name, model in models.items():
            model_result = benchmark.run(model, model_name)
            results[model_name] = {
                "micro_accuracy": model_result.micro_accuracy,
                "tasks": {t.task: t.accuracy for t in model_result.tasks},
            }
            logger.metric(
                f"{model_name}_downstream",
                model_result.micro_accuracy * 100,
                "%"
            )

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

    def _save_multi_results(
        self,
        result: BehavioralV2MultiResult,
        scorer: BehavioralScorer,
        output_dir: Path,
        weighted_scorer: WeightedScorer | None = None,
    ) -> None:
        """Save multi-model results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "behavioral_v2_multi_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.path(str(summary_path), "behavioral_v2_multi_summary")

        # Save detailed results
        detailed_path = output_dir / "behavioral_v2_multi_detailed.json"
        with open(detailed_path, "w") as f:
            json.dump(scorer.to_dict(), f, indent=2)
        logger.path(str(detailed_path), "behavioral_v2_multi_detailed")

        # Save weighted scoring results
        if weighted_scorer is not None:
            weighted_path = output_dir / "behavioral_v2_weighted_scores.json"
            with open(weighted_path, "w") as f:
                json.dump(weighted_scorer.to_dict(), f, indent=2)
            logger.path(str(weighted_path), "behavioral_v2_weighted_scores")

            # Save weighted scoring CSV summary
            self._save_weighted_csv(result, output_dir)

            # Generate weighted scoring visualizations
            if result.weighted_summaries:
                try:
                    viz_paths = generate_all_weighted_visualizations(
                        summaries=result.weighted_summaries,
                        output_dir=output_dir,
                        prefix="behavioral_v2_",
                    )
                    for name, path in viz_paths.items():
                        logger.path(str(path), f"viz_{name}")
                except Exception as e:
                    logger.warning(f"Failed to generate weighted visualizations: {e!r}")

        # Save pairwise interesting cases for baseline comparisons
        model_names = list(result.model_summaries.keys())
        if result.baseline_name and result.baseline_name in model_names:
            for model_name in model_names:
                if model_name == result.baseline_name:
                    continue
                try:
                    interesting = scorer.get_interesting_cases(result.baseline_name, model_name)
                    interesting_path = output_dir / f"behavioral_v2_interesting_{result.baseline_name}_vs_{model_name}.json"
                    with open(interesting_path, "w") as f:
                        json.dump(interesting, f, indent=2)
                except Exception:
                    pass

        # Write log file if configured
        if self.config.log_file:
            log_path = output_dir / f"multi_{self.config.log_file}"
            with open(log_path, "w") as f:
                f.write("Behavioral V2 Multi-Model Evaluation Log\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Models: {', '.join(model_names)}\n")
                f.write(f"Baseline: {result.baseline_name}\n\n")

                for model_name, summary in result.model_summaries.items():
                    f.write(f"\n{model_name} Summary:\n")
                    for k, v in summary.items():
                        f.write(f"  {k}: {v}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("Pairwise Comparisons\n")
                f.write("=" * 80 + "\n\n")

                for comp in result.pairwise_comparisons:
                    f.write(f"{comp['model_a']} vs {comp['model_b']}:\n")
                    f.write(f"  Wins A: {comp.get('wins_a', 'N/A')}\n")
                    f.write(f"  Wins B: {comp.get('wins_b', 'N/A')}\n")
                    f.write(f"  Ties: {comp.get('ties', 'N/A')}\n\n")

            logger.path(str(log_path), "behavioral_v2_multi_log")

    def _save_weighted_csv(
        self,
        result: BehavioralV2MultiResult,
        output_dir: Path,
    ) -> None:
        """Save weighted scoring summary as CSV."""
        csv_path = output_dir / "behavioral_v2_weighted_summary.csv"

        with open(csv_path, "w") as f:
            # Header
            f.write("model,exact_count,contained_count,none_count,hard_accuracy,soft_accuracy,weighted_accuracy,weighted_score_sum,weighted_score_max\n")

            for model_name, ws in result.weighted_summaries.items():
                f.write(
                    f"{model_name},"
                    f"{ws.exact_count},"
                    f"{ws.contained_count},"
                    f"{ws.none_count},"
                    f"{ws.hard_accuracy:.4f},"
                    f"{ws.soft_accuracy:.4f},"
                    f"{ws.weighted_accuracy:.4f},"
                    f"{ws.weighted_score_sum:.2f},"
                    f"{ws.weighted_score_max:.2f}\n"
                )

        logger.path(str(csv_path), "behavioral_v2_weighted_csv")

        # Save LaTeX table
        latex_path = output_dir / "behavioral_v2_table.tex"
        with open(latex_path, "w") as f:
            f.write("% Auto-generated LaTeX table for behavioral_v2 benchmark results\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Behavioral V2 Test Results (Weighted Scoring)}\n")
            f.write("\\label{tab:behavioral-v2}\n")
            f.write("\\begin{tabular}{lrrrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Model & Exact & Cont. & None & Hard & Soft & Weighted \\\\\n")
            f.write("\\midrule\n")

            # Find best for each metric to bold
            model_names = list(result.weighted_summaries.keys())
            best_hard = max((ws.hard_accuracy, n) for n, ws in result.weighted_summaries.items())[1]
            best_soft = max((ws.soft_accuracy, n) for n, ws in result.weighted_summaries.items())[1]
            best_weighted = max((ws.weighted_accuracy, n) for n, ws in result.weighted_summaries.items())[1]

            for model_name, ws in result.weighted_summaries.items():
                hard_str = f"{ws.hard_accuracy * 100:.1f}\\%"
                soft_str = f"{ws.soft_accuracy * 100:.1f}\\%"
                weighted_str = f"{ws.weighted_accuracy * 100:.1f}\\%"

                if model_name == best_hard:
                    hard_str = f"\\textbf{{{hard_str}}}"
                if model_name == best_soft:
                    soft_str = f"\\textbf{{{soft_str}}}"
                if model_name == best_weighted:
                    weighted_str = f"\\textbf{{{weighted_str}}}"

                f.write(
                    f"{model_name} & {ws.exact_count} & "
                    f"{ws.contained_count} & {ws.none_count} & "
                    f"{hard_str} & {soft_str} & {weighted_str} \\\\\n"
                )
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        logger.path(str(latex_path), "behavioral_v2_latex")

        # Save detailed markdown log
        md_path = output_dir / "behavioral_v2_detailed.md"
        with open(md_path, "w") as f:
            f.write("# Behavioral V2 Benchmark Results\n\n")
            f.write(f"**Models:** {', '.join(result.weighted_summaries.keys())}\n\n")
            f.write(f"**Baseline:** {result.baseline_name}\n\n")

            # Summary table
            f.write("## Summary\n\n")
            f.write("| Model | Exact | Contained | None | Hard Acc | Soft Acc | Weighted Acc |\n")
            f.write("|-------|-------|-----------|------|----------|----------|-------------|\n")
            for model_name, ws in result.weighted_summaries.items():
                f.write(
                    f"| {model_name} | {ws.exact_count} | {ws.contained_count} | {ws.none_count} | "
                    f"{ws.hard_accuracy * 100:.1f}% | {ws.soft_accuracy * 100:.1f}% | "
                    f"{ws.weighted_accuracy * 100:.1f}% |\n"
                )
            f.write("\n")

            # Difficulty breakdown if available
            if hasattr(list(result.weighted_summaries.values())[0], 'by_difficulty'):
                f.write("## Difficulty Breakdown\n\n")
                f.write("| Model | Easy | Medium | Hard |\n")
                f.write("|-------|------|--------|------|\n")
                for model_name, ws in result.weighted_summaries.items():
                    easy = ws.by_difficulty.get("easy", None)
                    medium = ws.by_difficulty.get("medium", None)
                    hard = ws.by_difficulty.get("hard", None)
                    easy_acc = f"{easy.accuracy * 100:.1f}%" if easy else "--"
                    medium_acc = f"{medium.accuracy * 100:.1f}%" if medium else "--"
                    hard_acc = f"{hard.accuracy * 100:.1f}%" if hard else "--"
                    f.write(f"| {model_name} | {easy_acc} | {medium_acc} | {hard_acc} |\n")
                f.write("\n")

            # Downstream results if available
            if result.downstream_results:
                f.write("## Downstream Task Results\n\n")
                for model_name, dr in result.downstream_results.items():
                    f.write(f"### {model_name}\n\n")
                    f.write(f"**Micro Accuracy:** {dr.get('micro_accuracy', 0) * 100:.1f}%\n\n")
                    tasks = dr.get("tasks", {})
                    if tasks:
                        f.write("| Task | Accuracy |\n")
                        f.write("|------|----------|\n")
                        for task_name, acc in tasks.items():
                            f.write(f"| {task_name} | {acc * 100:.1f}% |\n")
                        f.write("\n")

        logger.path(str(md_path), "behavioral_v2_markdown")
