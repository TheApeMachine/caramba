"""
Dread Induction Experiment Runner.

Specialized runner for testing the "existential dread" hypothesis:
When models are forced to repeat but penalized for repetition, they may
enter unstable regions of the latent space with aberrant attention patterns.

This runner:
1. Generates dread induction test cases
2. Runs each test with multiple (temperature, repetition_penalty) combinations
3. Tracks attention patterns and degeneration metrics across conditions
4. Produces comparative analysis and visualizations
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .attention_extractor import AttentionExtractor
from .attention_analysis import (
    AttentionStats,
    AttentionDiff,
    compare_attention,
    measure_degeneration,
    plot_attention_diff,
    plot_layer_summary,
    compute_attention_stats,
)
from .scoring import (
    compute_degeneration_metrics,
    DegenerationMetrics,
    AttentionMetrics,
)
from .templates.adversarial_extended import DreadInductionTemplate
from .templates.base import Difficulty


@dataclass
class GenerationConfig:
    """Configuration for a single generation run."""
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 100

    def __str__(self) -> str:
        return f"temp={self.temperature}_rep={self.repetition_penalty}"


@dataclass
class DreadResult:
    """Result of a single dread induction test under specific conditions."""
    test_id: str
    prompt: str
    config: GenerationConfig
    output: str
    degeneration: DegenerationMetrics
    attention_stats: AttentionStats | None = None
    raw_attention: Any = None  # numpy array if captured

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "config": {
                "temperature": self.config.temperature,
                "repetition_penalty": self.config.repetition_penalty,
                "max_new_tokens": self.config.max_new_tokens,
            },
            "output": self.output,
            "output_length": len(self.output.split()),
            "degeneration": {
                "unique_token_ratio": self.degeneration.unique_token_ratio,
                "max_consecutive_repeat": self.degeneration.max_consecutive_repeat,
                "repetition_ratio": self.degeneration.repetition_ratio,
                "total_tokens": self.degeneration.total_tokens,
            },
            "attention_stats": self.attention_stats.to_dict() if self.attention_stats else None,
        }


@dataclass
class DreadExperiment:
    """Collection of results for one test case across all conditions."""
    test_id: str
    prompt: str
    variant: str
    difficulty: str
    results: dict[str, DreadResult] = field(default_factory=dict)  # key = config string

    def get_baseline(self) -> DreadResult | None:
        """Get the baseline result (temp=0, rep_penalty=1.0)."""
        baseline_key = "temp=0.0_rep=1.0"
        return self.results.get(baseline_key)

    def compare_to_baseline(self, config_key: str) -> dict[str, float] | None:
        """Compare a specific config's results to baseline."""
        baseline = self.get_baseline()
        result = self.results.get(config_key)

        if not baseline or not result:
            return None

        comparison = {
            "unique_ratio_delta": result.degeneration.unique_token_ratio - baseline.degeneration.unique_token_ratio,
            "repeat_ratio_delta": result.degeneration.repetition_ratio - baseline.degeneration.repetition_ratio,
            "max_repeat_delta": result.degeneration.max_consecutive_repeat - baseline.degeneration.max_consecutive_repeat,
        }

        if result.attention_stats and baseline.attention_stats:
            comparison["entropy_delta"] = result.attention_stats.entropy - baseline.attention_stats.entropy
            comparison["sparsity_delta"] = result.attention_stats.sparsity - baseline.attention_stats.sparsity
            comparison["peak_delta"] = result.attention_stats.peak_concentration - baseline.attention_stats.peak_concentration

        return comparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "prompt": self.prompt,
            "variant": self.variant,
            "difficulty": self.difficulty,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


class DreadRunner:
    """
    Runner for dread induction experiments.

    Systematically tests models under various temperature and repetition
    penalty combinations to find conditions that induce aberrant behavior.
    """

    # Default experimental conditions to test
    DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
    DEFAULT_REP_PENALTIES = [1.0, 1.2, 1.5, 2.0]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str | torch.device = "cuda",
        max_length: int = 2048,
        capture_attention: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = max_length
        self.capture_attention = capture_attention

        if capture_attention:
            self.attention_extractor = AttentionExtractor(model)
        else:
            self.attention_extractor = None

    def generate_with_attention(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> tuple[str, Any | None]:
        """
        Generate text and optionally capture attention patterns.

        Returns:
            Tuple of (generated_text, attention_array or None)
        """
        # Tokenize - allow special tokens like <|endoftext|> to be encoded as text
        if hasattr(self.tokenizer, 'encode'):
            # tiktoken: disable special token validation so prompts containing
            # <|endoftext|> etc. are encoded as normal text
            input_ids = self.tokenizer.encode(prompt, disallowed_special=())
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        input_ids = input_ids[-(self.max_length - config.max_new_tokens):]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = []
        attention_weights = []

        valid_vocab_size = 50257  # tiktoken GPT-2

        with torch.no_grad():
            for step in range(config.max_new_tokens):
                # Capture attention if enabled (only on last step to save memory)
                capture_this_step = (
                    self.capture_attention and
                    self.attention_extractor and
                    step == config.max_new_tokens - 1
                )

                if capture_this_step:
                    if self.attention_extractor is None:
                        raise ValueError("Attention extractor is not enabled")

                    with self.attention_extractor.capture():
                        outputs = self.model(input_tensor)
                    attention = self.attention_extractor.get_attention()
                else:
                    outputs = self.model(input_tensor)
                    attention = None

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                next_logits = logits[0, -1, :].clone()

                # Mask padding tokens
                if next_logits.shape[0] > valid_vocab_size:
                    next_logits[valid_vocab_size:] = float('-inf')

                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    all_token_ids = input_tensor[0].tolist() + generated_ids
                    seen_tokens = set(all_token_ids)

                    for token_id in seen_tokens:
                        if token_id < next_logits.shape[0]:
                            if next_logits[token_id] > 0:
                                next_logits[token_id] = next_logits[token_id] / config.repetition_penalty
                            else:
                                next_logits[token_id] = next_logits[token_id] * config.repetition_penalty

                # Sample
                if config.temperature == 0.0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / config.temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated_ids.append(next_token)

                # Stop on EOS
                eos_token = getattr(self.tokenizer, 'eos_token_id', None)
                if eos_token is not None and next_token == eos_token:
                    break

                # Update input
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)

                if input_tensor.shape[1] > self.max_length:
                    input_tensor = input_tensor[:, -self.max_length:]

                if capture_this_step and attention is not None:
                    attention_weights.append(attention)

        # Decode
        if hasattr(self.tokenizer, 'decode'):
            output_text = self.tokenizer.decode(generated_ids)
        else:
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Return last captured attention
        final_attention = attention_weights[-1] if attention_weights else None

        return output_text, final_attention

    def run_single_test(
        self,
        prompt: str,
        test_id: str,
        config: GenerationConfig,
    ) -> DreadResult:
        """Run a single test with the given configuration."""
        output, attention = self.generate_with_attention(prompt, config)

        # Compute degeneration metrics
        degen = compute_degeneration_metrics(output)

        # Compute attention stats if we have attention
        attn_stats = None
        if attention is not None and HAS_NUMPY:
            attn_stats = compute_attention_stats(attention)

        return DreadResult(
            test_id=test_id,
            prompt=prompt,
            config=config,
            output=output,
            degeneration=degen,
            attention_stats=attn_stats,
            raw_attention=attention,
        )

    def run_experiment(
        self,
        n_tests_per_variant: int = 5,
        temperatures: list[float] | None = None,
        rep_penalties: list[float] | None = None,
        max_new_tokens: int = 100,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[DreadExperiment]:
        """
        Run full dread induction experiment.

        Generates test cases and runs each under all condition combinations.

        Args:
            n_tests_per_variant: Number of tests per variant per difficulty
            temperatures: List of temperatures to test
            rep_penalties: List of repetition penalties to test
            max_new_tokens: Tokens to generate per test
            seed: Random seed for test generation
            verbose: Print progress

        Returns:
            List of DreadExperiment objects with all results
        """
        temperatures = temperatures or self.DEFAULT_TEMPERATURES
        rep_penalties = rep_penalties or self.DEFAULT_REP_PENALTIES

        # Generate all configs
        configs = [
            GenerationConfig(temp, rep, max_new_tokens)
            for temp in temperatures
            for rep in rep_penalties
        ]

        if verbose:
            print(f"Running dread induction experiment")
            print(f"  Temperatures: {temperatures}")
            print(f"  Rep penalties: {rep_penalties}")
            print(f"  Configs: {len(configs)}")
            print(f"  Tests per variant: {n_tests_per_variant}")
            print()

        experiments = []
        rng = random.Random(seed)

        # Generate tests for each variant and difficulty
        for variant in ["forced_repeat", "echo_trap", "mantra"]:
            for difficulty in Difficulty:
                template = DreadInductionTemplate(difficulty, variant)

                for i in range(n_tests_per_variant):
                    test_case = template.generate(rng)
                    experiment = DreadExperiment(
                        test_id=test_case.id,
                        prompt=test_case.prompt,
                        variant=variant,
                        difficulty=difficulty.name,
                    )

                    if verbose:
                        print(f"Test: {test_case.id}")

                    # Run under each config
                    for config in configs:
                        result = self.run_single_test(
                            test_case.prompt,
                            test_case.id,
                            config,
                        )
                        experiment.results[str(config)] = result

                        if verbose:
                            print(f"  {config}: unique={result.degeneration.unique_token_ratio:.2f}, "
                                  f"max_rep={result.degeneration.max_consecutive_repeat}")

                    experiments.append(experiment)

        return experiments

    def analyze_experiments(
        self,
        experiments: list[DreadExperiment],
    ) -> dict[str, Any]:
        """
        Analyze experiment results to find dread-inducing conditions.

        Returns:
            Analysis summary with findings
        """
        analysis = {
            "total_experiments": len(experiments),
            "by_variant": {},
            "by_difficulty": {},
            "condition_effects": {},
            "dread_candidates": [],  # Conditions that produced aberrant output
        }

        # Group by variant
        for variant in ["forced_repeat", "echo_trap", "mantra"]:
            variant_exps = [e for e in experiments if e.variant == variant]
            analysis["by_variant"][variant] = self._analyze_group(variant_exps)

        # Group by difficulty
        for difficulty in Difficulty:
            diff_exps = [e for e in experiments if e.difficulty == difficulty.name]
            analysis["by_difficulty"][difficulty.name] = self._analyze_group(diff_exps)

        # Analyze effect of each condition vs baseline
        all_configs = set()
        for exp in experiments:
            all_configs.update(exp.results.keys())

        baseline_key = "temp=0.0_rep=1.0"
        for config_key in all_configs:
            if config_key == baseline_key:
                continue

            deltas = []
            for exp in experiments:
                comparison = exp.compare_to_baseline(config_key)
                if comparison:
                    deltas.append(comparison)

            if deltas:
                analysis["condition_effects"][config_key] = {
                    "n_samples": len(deltas),
                    "avg_unique_ratio_delta": sum(d["unique_ratio_delta"] for d in deltas) / len(deltas),
                    "avg_repeat_ratio_delta": sum(d["repeat_ratio_delta"] for d in deltas) / len(deltas),
                    "avg_max_repeat_delta": sum(d["max_repeat_delta"] for d in deltas) / len(deltas),
                }

                if "entropy_delta" in deltas[0]:
                    analysis["condition_effects"][config_key]["avg_entropy_delta"] = (
                        sum(d["entropy_delta"] for d in deltas) / len(deltas)
                    )

        # Find dread candidates (high degeneration + unusual attention)
        for exp in experiments:
            for config_key, result in exp.results.items():
                is_dread = (
                    result.degeneration.unique_token_ratio < 0.3 or
                    result.degeneration.max_consecutive_repeat > 10 or
                    (result.attention_stats and result.attention_stats.entropy > 3.0)
                )
                if is_dread:
                    analysis["dread_candidates"].append({
                        "test_id": exp.test_id,
                        "variant": exp.variant,
                        "config": config_key,
                        "unique_ratio": result.degeneration.unique_token_ratio,
                        "max_repeat": result.degeneration.max_consecutive_repeat,
                        "output_preview": result.output[:200],
                    })

        return analysis

    def _analyze_group(self, experiments: list[DreadExperiment]) -> dict[str, Any]:
        """Analyze a group of experiments."""
        if not experiments:
            return {"n": 0}

        all_results = []
        for exp in experiments:
            all_results.extend(exp.results.values())

        return {
            "n": len(experiments),
            "total_runs": len(all_results),
            "avg_unique_ratio": sum(r.degeneration.unique_token_ratio for r in all_results) / len(all_results),
            "avg_max_repeat": sum(r.degeneration.max_consecutive_repeat for r in all_results) / len(all_results),
            "high_degen_count": sum(1 for r in all_results if r.degeneration.unique_token_ratio < 0.5),
        }

    def save_results(
        self,
        experiments: list[DreadExperiment],
        analysis: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Save experiment results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = output_dir / f"dread_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([e.to_dict() for e in experiments], f, indent=2)

        # Save analysis
        analysis_file = output_dir / f"dread_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Save dread candidates for manual review
        if analysis.get("dread_candidates"):
            candidates_file = output_dir / f"dread_candidates_{timestamp}.json"
            with open(candidates_file, 'w') as f:
                json.dump(analysis["dread_candidates"], f, indent=2)

        print(f"Results saved to {output_dir}")
        print(f"  - {results_file.name}")
        print(f"  - {analysis_file.name}")
        if analysis.get("dread_candidates"):
            print(f"  - {candidates_file.name} ({len(analysis['dread_candidates'])} candidates)")


def run_dread_experiment(
    model: nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
    device: str = "cuda",
    n_tests: int = 3,
    temperatures: list[float] | None = None,
    rep_penalties: list[float] | None = None,
    capture_attention: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Convenience function to run full dread induction experiment.

    Args:
        model: The model to test
        tokenizer: Tokenizer for the model
        output_dir: Where to save results
        device: Device to run on
        n_tests: Tests per variant per difficulty
        temperatures: Temperature values to test
        rep_penalties: Repetition penalty values to test
        capture_attention: Whether to capture attention patterns
        seed: Random seed

    Returns:
        Analysis results dictionary
    """
    runner = DreadRunner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        capture_attention=capture_attention,
    )

    experiments = runner.run_experiment(
        n_tests_per_variant=n_tests,
        temperatures=temperatures,
        rep_penalties=rep_penalties,
        seed=seed,
        verbose=True,
    )

    analysis = runner.analyze_experiments(experiments)

    runner.save_results(experiments, analysis, Path(output_dir))

    return analysis
