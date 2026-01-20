"""
Multi-model evaluation runner with attention capture.

Runs tests across multiple models and captures:
- Raw text outputs
- Token logprobs (for choice tasks)
- Attention weights (for visualization)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
from tqdm import tqdm

from .templates.base import TestCase, EvalKind
from .scoring import ModelOutput, TestScore, MultiModelScorer, SoftScore


class ModelProtocol(Protocol):
    """Protocol for models that can be evaluated."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from prompt."""
        ...

    def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        """Get log probabilities for each choice."""
        ...

    def get_attention_weights(
        self,
        prompt: str,
    ) -> np.ndarray | None:
        """Get attention weights. Shape: [layers, heads, seq, seq]"""
        ...


@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    max_new_tokens: int = 16
    temperature: float = 0.0
    capture_attention: bool = True
    attention_layers: list[int] | None = None  # None = all layers
    batch_size: int = 1  # For future batched evaluation
    show_progress: bool = True
    save_raw_outputs: bool = True


@dataclass
class EvalResults:
    """Complete evaluation results."""
    config: EvalConfig
    model_ids: list[str]
    test_count: int
    scores: dict[str, dict[str, TestScore]]  # model_id -> test_id -> score
    summaries: dict[str, dict[str, Any]]      # model_id -> summary stats
    comparisons: list[dict[str, Any]]         # Head-to-head comparisons
    category_results: dict[str, dict[str, dict[str, Any]]]  # category -> model -> stats
    attention_data: dict[str, dict[str, np.ndarray]] | None  # model -> test -> attention

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict (excluding attention data)."""
        return {
            "config": {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "capture_attention": self.config.capture_attention,
            },
            "model_ids": self.model_ids,
            "test_count": self.test_count,
            "summaries": self.summaries,
            "comparisons": self.comparisons,
            "category_results": self.category_results,
            "scores": {
                model_id: {
                    test_id: {
                        "exact_match": score.exact_match,
                        "content_match": score.content_match,
                        "soft_score": score.soft_score.value,
                        "soft_notes": score.soft_notes,
                        "expected": score.expected,
                        "actual": score.actual,
                        "flags": {
                            "repetition_loop": score.flags.repetition_loop,
                            "distractor_contamination": score.flags.distractor_contamination,
                            "empty_output": score.flags.empty_output,
                            "format_continuation": score.flags.format_continuation,
                        },
                    }
                    for test_id, score in tests.items()
                }
                for model_id, tests in self.scores.items()
            },
        }

    def save(self, output_dir: Path) -> None:
        """Save results to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results as JSON
        with open(output_dir / "results.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save attention data as compressed numpy
        if self.attention_data:
            for model_id, tests in self.attention_data.items():
                model_dir = output_dir / "attention" / model_id
                model_dir.mkdir(parents=True, exist_ok=True)
                for test_id, attn in tests.items():
                    np.savez_compressed(
                        model_dir / f"{test_id}.npz",
                        attention=attn,
                    )


class EvalRunner:
    """
    Runs behavioral evaluation across multiple models.
    """

    def __init__(
        self,
        models: dict[str, ModelProtocol],
        config: EvalConfig | None = None,
    ):
        """
        Initialize runner with models to evaluate.

        Args:
            models: Dict mapping model_id to model instance
            config: Evaluation configuration
        """
        self.models = models
        self.config = config or EvalConfig()
        self.scorer = MultiModelScorer(list(models.keys()))

    def run(self, tests: list[TestCase]) -> EvalResults:
        """
        Run evaluation on all tests across all models.

        Args:
            tests: List of test cases to evaluate

        Returns:
            Complete EvalResults with all metrics
        """
        attention_data: dict[str, dict[str, np.ndarray]] = {
            mid: {} for mid in self.models
        }

        # Create progress bar
        total_evals = len(tests) * len(self.models)
        pbar = tqdm(
            total=total_evals,
            desc="Evaluating",
            disable=not self.config.show_progress,
        )

        # Run each test on each model
        for test in tests:
            for model_id, model in self.models.items():
                output = self._run_single(model_id, model, test)

                # Score the output
                self.scorer.add_result(
                    output=output,
                    expected=str(test.expected),
                    prompt=test.prompt,
                    choices=test.choices,
                )

                # Store attention if captured
                if output.attention_weights is not None:
                    attention_data[model_id][test.id] = output.attention_weights

                pbar.update(1)

        pbar.close()

        # Compute summaries and comparisons
        summaries = {mid: self.scorer.get_summary(mid) for mid in self.models}
        comparisons = self.scorer.get_all_comparisons()

        # Compute per-category results
        category_results = self._compute_category_results(tests)

        return EvalResults(
            config=self.config,
            model_ids=list(self.models.keys()),
            test_count=len(tests),
            scores=self.scorer.scores,
            summaries=summaries,
            comparisons=comparisons,
            category_results=category_results,
            attention_data=attention_data if self.config.capture_attention else None,
        )

    def _run_single(
        self,
        model_id: str,
        model: ModelProtocol,
        test: TestCase,
    ) -> ModelOutput:
        """Run a single test on a single model."""
        start_time = time.time()

        # Generate output based on test kind
        if test.kind == EvalKind.CHOICE_LOGPROB:
            # For choice tasks, score via logprob argmax (generation is unnecessary and can be misleading).
            choices = test.choices or []
            logprobs = model.get_choice_logprobs(test.prompt, choices) if choices else {}
            if logprobs:
                output_text = max(logprobs.items(), key=lambda kv: kv[1])[0]
            else:
                output_text = ""
        else:
            logprobs = None
            output_text = model.generate(
                test.prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

        # Capture attention if configured
        attention = None
        if self.config.capture_attention:
            attention = model.get_attention_weights(test.prompt)

        elapsed_ms = (time.time() - start_time) * 1000

        return ModelOutput(
            model_id=model_id,
            test_id=test.id,
            output_text=output_text,
            logprobs=logprobs,
            attention_weights=attention,
            generation_time_ms=elapsed_ms,
        )

    def _compute_category_results(
        self,
        tests: list[TestCase],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute per-category statistics for each model."""
        # Group tests by category
        by_category: dict[str, list[str]] = {}
        for test in tests:
            if test.category not in by_category:
                by_category[test.category] = []
            by_category[test.category].append(test.id)

        results = {}
        for category, test_ids in by_category.items():
            results[category] = {}
            for model_id in self.models:
                scores = [
                    self.scorer.scores[model_id][tid]
                    for tid in test_ids
                    if tid in self.scorer.scores[model_id]
                ]

                if not scores:
                    continue

                n = len(scores)
                results[category][model_id] = {
                    "count": n,
                    "exact_match_rate": sum(1 for s in scores if s.exact_match) / n,
                    "content_match_rate": sum(1 for s in scores if s.content_match) / n,
                    "soft_score_avg": sum(s.soft_score for s in scores) / n,
                    "score_distribution": {
                        score.name: sum(1 for s in scores if s.soft_score == score)
                        for score in SoftScore
                    },
                }

        return results


class MockModel:
    """Mock model for testing the evaluation framework."""

    def __init__(self, model_id: str, accuracy: float = 0.8):
        self.model_id = model_id
        self.accuracy = accuracy
        import random
        self.rng = random.Random(hash(model_id))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """Generate mock output."""
        # Extract expected from simple copy prompts
        if "Copy:" in prompt:
            lines = prompt.strip().split('\n')
            # Find the last non-empty line that has content after Copy:
            for line in reversed(lines[:-1]):
                if "Copy:" in line and line.split("Copy:")[-1].strip():
                    expected = line.split("Copy:")[-1].strip()
                    if self.rng.random() < self.accuracy:
                        return expected
                    else:
                        return "WRONG"
        return "UNKNOWN"

    def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        """Generate mock logprobs."""
        logprobs = {}
        for i, choice in enumerate(choices):
            # First choice has highest prob if "accurate"
            if i == 0 and self.rng.random() < self.accuracy:
                logprobs[choice] = -0.1
            else:
                logprobs[choice] = -2.0 - i
        return logprobs

    def get_attention_weights(self, prompt: str) -> np.ndarray | None:
        """Generate mock attention weights."""
        # Return None to skip attention capture in tests
        return None


if __name__ == "__main__":
    # Demo with mock models
    from .generator import generate_suite

    # Generate small test suite
    suite = generate_suite(seed=42, tests_per_category=5)

    # Create mock models
    models = {
        "baseline": MockModel("baseline", accuracy=0.85),
        "dba": MockModel("dba", accuracy=0.80),
    }

    # Run evaluation
    config = EvalConfig(capture_attention=False, show_progress=True)
    runner = EvalRunner(models, config)
    results = runner.run(suite.tests)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for model_id, summary in results.summaries.items():
        print(f"\n{model_id}:")
        print(f"  Exact match rate: {summary['exact_match_rate']:.1%}")
        print(f"  Soft score avg: {summary['soft_score_avg']:.2f}")
        print(f"  Score distribution: {summary['score_distribution']}")

    print("\nHead-to-head comparisons:")
    for comp in results.comparisons:
        print(f"  {comp['model_a']} vs {comp['model_b']}:")
        print(f"    Wins A: {comp['wins_a']}, Wins B: {comp['wins_b']}, Ties: {comp['ties']}")
