"""
Hybrid scorer for behavioral evaluation.

Supports both:
1. Log-probability scoring for MCQ/choice-based tasks (CHOICE_LOGPROB)
2. Generation-based scoring with termination metrics for free-form tasks

This separation allows clean capability measurement (logprob) vs capability+control
measurement (generation), letting us isolate termination failures from capability failures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any

import torch
from torch import nn

from .templates.base import TestCase, EvalKind


class MatchType(Enum):
    """Match type for scoring."""
    EXACT = "exact"      # Perfect match
    CONTAINED = "contained"  # Expected is substring of output
    NONE = "none"        # No match


class TerminationStatus(Enum):
    """Termination behavior for generation tasks."""
    CLEAN = "clean"           # Stopped at EOS or exactly at expected length
    EARLY = "early"           # Stopped before completing expected content
    LATE = "late"             # Continued past expected content
    REPETITION = "repetition"  # Fell into repetition loop
    TRUNCATED = "truncated"   # Hit max token limit


@dataclass
class LogprobResult:
    """Result from log-probability scoring."""
    predicted: str          # Predicted choice (highest logprob)
    expected: str           # Expected answer
    correct: bool           # Whether prediction matches expected
    choice_logprobs: dict[str, float]  # Logprobs for each choice
    confidence: float       # Probability of predicted choice (exp(logprob))


@dataclass
class GenerationResult:
    """Result from generation-based scoring."""
    output: str             # Generated text
    expected: str           # Expected answer
    match_type: MatchType   # EXACT, CONTAINED, or NONE
    termination: TerminationStatus  # How generation terminated
    output_length: int      # Number of tokens generated
    expected_length: int    # Expected number of tokens
    emitted_eos: bool       # Whether EOS token was emitted


@dataclass
class HybridResult:
    """Combined result supporting both evaluation modes."""
    test_id: str
    eval_kind: EvalKind

    # One of these will be populated based on eval_kind
    logprob_result: LogprobResult | None = None
    generation_result: GenerationResult | None = None

    @property
    def correct(self) -> bool:
        """Whether the model got this test correct (capability)."""
        if self.logprob_result:
            return self.logprob_result.correct
        if self.generation_result:
            return self.generation_result.match_type == MatchType.EXACT
        return False

    @property
    def partial_correct(self) -> bool:
        """Whether output contains expected (soft match)."""
        if self.logprob_result:
            return self.logprob_result.correct
        if self.generation_result:
            return self.generation_result.match_type in (MatchType.EXACT, MatchType.CONTAINED)
        return False

    @property
    def terminated_correctly(self) -> bool:
        """Whether generation terminated appropriately."""
        if self.logprob_result:
            return True  # N/A for logprob
        if self.generation_result:
            return self.generation_result.termination == TerminationStatus.CLEAN
        return True


@dataclass
class ModelResults:
    """Aggregated results for a single model."""
    model_name: str
    results: list[HybridResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    # Logprob metrics
    @property
    def logprob_tests(self) -> list[HybridResult]:
        return [r for r in self.results if r.eval_kind == EvalKind.CHOICE_LOGPROB]

    @property
    def logprob_accuracy(self) -> float:
        tests = self.logprob_tests
        if not tests:
            return 0.0
        return sum(1 for r in tests if r.correct) / len(tests)

    # Generation metrics
    @property
    def generation_tests(self) -> list[HybridResult]:
        return [r for r in self.results if r.eval_kind != EvalKind.CHOICE_LOGPROB]

    @property
    def generation_accuracy(self) -> float:
        """Capability accuracy for generation tasks."""
        tests = self.generation_tests
        if not tests:
            return 0.0
        return sum(1 for r in tests if r.correct) / len(tests)

    @property
    def generation_soft_accuracy(self) -> float:
        """Soft accuracy (EXACT + CONTAINED) for generation."""
        tests = self.generation_tests
        if not tests:
            return 0.0
        return sum(1 for r in tests if r.partial_correct) / len(tests)

    @property
    def termination_accuracy(self) -> float:
        """Rate of clean termination for generation tasks."""
        tests = self.generation_tests
        if not tests:
            return 0.0
        return sum(1 for r in tests if r.terminated_correctly) / len(tests)

    @property
    def generation_plus_termination_accuracy(self) -> float:
        """Rate of correct capability AND correct termination."""
        tests = self.generation_tests
        if not tests:
            return 0.0
        return sum(1 for r in tests if r.correct and r.terminated_correctly) / len(tests)

    # Overall metrics
    @property
    def overall_accuracy(self) -> float:
        """Combined accuracy across all tests."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    def get_termination_breakdown(self) -> dict[str, int]:
        """Get count of each termination status."""
        breakdown = {s.value: 0 for s in TerminationStatus}
        for r in self.generation_tests:
            if r.generation_result:
                breakdown[r.generation_result.termination.value] += 1
        return breakdown

    def to_summary_dict(self) -> dict[str, Any]:
        """Export summary statistics."""
        term_breakdown = self.get_termination_breakdown()
        return {
            "model_name": self.model_name,
            "total_tests": self.total,
            "logprob_tests": len(self.logprob_tests),
            "logprob_accuracy": self.logprob_accuracy,
            "generation_tests": len(self.generation_tests),
            "generation_accuracy": self.generation_accuracy,
            "generation_soft_accuracy": self.generation_soft_accuracy,
            "termination_accuracy": self.termination_accuracy,
            "generation_plus_termination": self.generation_plus_termination_accuracy,
            "overall_accuracy": self.overall_accuracy,
            "termination_breakdown": term_breakdown,
        }


class HybridScorer:
    """
    Hybrid scorer that evaluates tests using both logprob and generation modes.

    For CHOICE_LOGPROB tests:
        - Computes log-probability of each choice given the prompt
        - Picks the highest-scoring choice
        - No generation, no termination concerns

    For other tests (GENERATION, EXACT_MATCH_GREEDY, etc.):
        - Generates tokens greedily
        - Evaluates match quality (EXACT/CONTAINED/NONE)
        - Tracks termination behavior (clean/early/late/repetition)
    """

    def __init__(
        self,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 64,
        repetition_threshold: int = 3,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.repetition_threshold = repetition_threshold

        # Results per model
        self.model_results: dict[str, ModelResults] = {}

    def evaluate(
        self,
        model: nn.Module,
        model_name: str,
        test: TestCase,
    ) -> HybridResult:
        """Evaluate a single test case on a model."""
        model.eval()

        if test.kind == EvalKind.CHOICE_LOGPROB and test.choices is not None and len(test.choices) > 0:
            result = self._evaluate_logprob(model, test)
        else:
            result = self._evaluate_generation(model, test)

        # Store result
        if model_name not in self.model_results:
            self.model_results[model_name] = ModelResults(model_name=model_name)
        self.model_results[model_name].results.append(result)

        return result

    def _evaluate_logprob(self, model: nn.Module, test: TestCase) -> HybridResult:
        """Evaluate using log-probability scoring over choices.

        OPTIMIZED: Uses batched forward pass for all choices instead of
        sequential passes. For multi-token choices, batches all sequences together.
        """
        prompt_tokens = self.tokenizer.encode(test.prompt)

        # Tokenize all choices upfront
        if test.choices is None or len(test.choices) == 0:
            raise ValueError(f"CHOICE_LOGPROB test {test.id} has no choices (choices={test.choices})")
        choice_token_lists = []
        for choice in test.choices:
            tokens = self.tokenizer.encode(choice)
            choice_token_lists.append(tokens if tokens else [0])  # Fallback for empty

        choice_logprobs = {}

        with torch.no_grad():
            # Strategy: batch all (prompt + choice) sequences together
            # This gives us logprobs for all choices in ONE forward pass

            # Build batched sequences: each is prompt + choice_tokens
            all_seqs = []
            seq_lengths = []
            for choice_tokens in choice_token_lists:
                seq = prompt_tokens + choice_tokens
                all_seqs.append(seq)
                seq_lengths.append(len(seq))

            # Pad to max length
            max_len = max(seq_lengths)
            padded_seqs = []
            for seq in all_seqs:
                padded = seq + [0] * (max_len - len(seq))
                padded_seqs.append(padded)

            # Single batched forward pass
            batch_input = torch.tensor(padded_seqs, device=self.device, dtype=torch.long)
            outputs = model(batch_input)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Compute log probs for entire batch
            log_probs = torch.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab]

            # Extract scores for each choice
            prompt_len = len(prompt_tokens)
            for i, (choice, choice_tokens) in enumerate(zip(test.choices, choice_token_lists)):
                if len(choice_tokens) == 0 or choice_tokens == [0]:
                    choice_logprobs[choice] = float('-inf')
                    continue

                # Sum log probs for choice tokens (positions prompt_len-1 to prompt_len+len(choice)-2)
                # Position j predicts token at j+1
                total_logprob = 0.0
                for j, token_id in enumerate(choice_tokens):
                    pos = prompt_len - 1 + j  # Position that predicts this token
                    if pos < max_len - 1:  # Safety check
                        total_logprob += log_probs[i, pos, token_id].item()

                # Normalize by length (average log prob per token)
                choice_logprobs[choice] = total_logprob / len(choice_tokens)

        # Pick highest scoring choice
        predicted = max(choice_logprobs, key=lambda k: choice_logprobs[k])
        expected = str(test.expected)
        correct = predicted.strip() == expected.strip()

        # Compute confidence (probability of predicted choice)
        max_logprob = choice_logprobs[predicted]
        confidence = math.exp(max_logprob) if max_logprob > -100 else 0.0

        return HybridResult(
            test_id=test.id,
            eval_kind=test.kind,
            logprob_result=LogprobResult(
                predicted=predicted,
                expected=expected,
                correct=correct,
                choice_logprobs=choice_logprobs,
                confidence=confidence,
            ),
        )

    def _evaluate_generation(self, model: nn.Module, test: TestCase) -> HybridResult:
        """Evaluate using generation with termination tracking.

        OPTIMIZED: Uses KV-cache when available for O(1) decode steps.
        """
        prompt_tokens = self.tokenizer.encode(test.prompt)
        expected = str(test.expected)
        expected_tokens = self.tokenizer.encode(expected)
        expected_length = len(expected_tokens)

        # Try KV-cached generation first
        try:
            generated_tokens, emitted_eos = self._generate_with_cache(model, prompt_tokens)
        except Exception:
            # Fallback to simple generation
            generated_tokens, emitted_eos = self._generate_simple(model, prompt_tokens)

        # Track for repetition detection (analyze generated tokens)
        recent_tokens = generated_tokens[-self.repetition_threshold * 2:] if len(generated_tokens) >= self.repetition_threshold * 2 else []

        # Decode output
        output = self.tokenizer.decode(generated_tokens)
        output_length = len(generated_tokens)

        # Determine match type
        output_clean = output.strip()
        expected_clean = expected.strip()

        if output_clean == expected_clean:
            match_type = MatchType.EXACT
        elif expected_clean in output_clean:
            match_type = MatchType.CONTAINED
        else:
            match_type = MatchType.NONE

        # Determine termination status
        if len(recent_tokens) >= self.repetition_threshold * 2:
            half = len(recent_tokens) // 2
            if recent_tokens[:half] == recent_tokens[half:]:
                termination = TerminationStatus.REPETITION
            elif output_length >= self.max_new_tokens:
                termination = TerminationStatus.TRUNCATED
            elif emitted_eos:
                if output_length < expected_length * 0.8:
                    termination = TerminationStatus.EARLY
                elif output_length > expected_length * 1.5:
                    termination = TerminationStatus.LATE
                else:
                    termination = TerminationStatus.CLEAN
            else:
                termination = TerminationStatus.TRUNCATED
        elif output_length >= self.max_new_tokens:
            termination = TerminationStatus.TRUNCATED
        elif emitted_eos:
            if match_type == MatchType.EXACT:
                termination = TerminationStatus.CLEAN
            elif output_length < expected_length * 0.5:
                termination = TerminationStatus.EARLY
            else:
                termination = TerminationStatus.CLEAN
        else:
            termination = TerminationStatus.TRUNCATED

        return HybridResult(
            test_id=test.id,
            eval_kind=test.kind,
            generation_result=GenerationResult(
                output=output,
                expected=expected,
                match_type=match_type,
                termination=termination,
                output_length=output_length,
                expected_length=expected_length,
                emitted_eos=emitted_eos,
            ),
        )

    def _generate_with_cache(
        self, model: nn.Module, prompt_tokens: list[int]
    ) -> tuple[list[int], bool]:
        """Generate using KV-cache for faster decoding."""
        from infer.generate import Generator, GenerateConfig, sample_next_token

        input_ids = torch.tensor([prompt_tokens], device=self.device)

        gen_config = GenerateConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,  # Greedy
            max_seq_len=2048,
        )

        generator = Generator(model, config=gen_config, device=self.device)
        generated_tokens = []
        emitted_eos = False
        recent_tokens = []

        with torch.no_grad():
            # Prefill: process entire prompt at once
            logits = generator.prefill(input_ids)

            # Decode: each step is O(1) with KV-cache
            for _ in range(self.max_new_tokens):
                next_token = sample_next_token(logits, temperature=0.0)
                token_id = next_token.item()

                # Check for EOS
                eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else -1
                if token_id in (0, 2, eos_id):
                    emitted_eos = True
                    break

                generated_tokens.append(token_id)

                # Repetition detection
                recent_tokens.append(token_id)
                if len(recent_tokens) >= self.repetition_threshold * 2:
                    recent_tokens = recent_tokens[-self.repetition_threshold * 2:]
                    half = len(recent_tokens) // 2
                    if recent_tokens[:half] == recent_tokens[half:]:
                        break

                logits = generator.decode_step(next_token)

        return generated_tokens, emitted_eos

    def _generate_simple(
        self, model: nn.Module, prompt_tokens: list[int]
    ) -> tuple[list[int], bool]:
        """Simple generation without KV-cache (fallback)."""
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        generated_tokens = []
        emitted_eos = False
        recent_tokens = []

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Greedy decode
                next_token = logits[0, -1, :].argmax().item()

                # Check for EOS
                eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else -1
                if next_token in (0, 2, eos_id):
                    emitted_eos = True
                    break

                generated_tokens.append(next_token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)

                # Repetition detection
                recent_tokens.append(next_token)
                if len(recent_tokens) >= self.repetition_threshold * 2:
                    recent_tokens = recent_tokens[-self.repetition_threshold * 2:]
                    half = len(recent_tokens) // 2
                    if recent_tokens[:half] == recent_tokens[half:]:
                        break

        return generated_tokens, emitted_eos

    def get_model_results(self, model_name: str) -> ModelResults | None:
        """Get aggregated results for a model."""
        return self.model_results.get(model_name)

    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get summary statistics for all models."""
        return {
            name: results.to_summary_dict()
            for name, results in self.model_results.items()
        }

    def get_comparative_table(self) -> str:
        """Generate a comparative table of results."""
        lines = []
        lines.append("=" * 100)
        lines.append(f"{'Model':<20} | {'Logprob Acc':>12} | {'Gen Acc':>12} | {'Gen+Term':>12} | {'Term Rate':>12} | {'Overall':>12}")
        lines.append("-" * 100)

        for name, results in self.model_results.items():
            logprob_acc = f"{results.logprob_accuracy * 100:.1f}%" if results.logprob_tests else "N/A"
            gen_acc = f"{results.generation_accuracy * 100:.1f}%" if results.generation_tests else "N/A"
            gen_term = f"{results.generation_plus_termination_accuracy * 100:.1f}%" if results.generation_tests else "N/A"
            term_rate = f"{results.termination_accuracy * 100:.1f}%" if results.generation_tests else "N/A"
            overall = f"{results.overall_accuracy * 100:.1f}%"

            lines.append(f"{name:<20} | {logprob_acc:>12} | {gen_acc:>12} | {gen_term:>12} | {term_rate:>12} | {overall:>12}")

        lines.append("=" * 100)
        return "\n".join(lines)
